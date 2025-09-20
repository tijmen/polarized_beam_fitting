"""
Bootstrap uncertainty estimation for polarized beam fitting.

 - Warm-start from non-resampled best-fit parameters
 - No closure over the data (JAX doesn't compile the data in as constants)
 - Parallel execution across multiple GPUs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .fitter import ObjectiveFunctions, PolarizedBeamFitter
from .utils import params_from_logit, params_to_logit


def mask_gradients_for_excluded_sources(grad_tree, weight_array):
    """
    Zero out gradients for source parameters where weight = 0.
    Beam parameters are always kept since they affect all sources.
    """
    zero_weight_mask = weight_array == 0

    # Create a copy of the gradient tree
    masked_grads = jax.tree.map(lambda x: x, grad_tree)

    # Mask source parameter gradients for excluded sources
    if "sources" in masked_grads:
        for param_name in ["yoff", "xoff", "flux"]:
            if param_name in masked_grads["sources"]:
                if param_name == "flux":
                    # For flux gradients, need to expand mask to match shape (n_sources, n_bands, 3)
                    flux_grad_shape = masked_grads["sources"][param_name].shape
                    zero_weight_mask_expanded = jnp.broadcast_to(zero_weight_mask.reshape(-1, 1, 1), flux_grad_shape)
                    masked_grads["sources"][param_name] = jnp.where(
                        zero_weight_mask_expanded,
                        jnp.zeros_like(masked_grads["sources"][param_name]),
                        masked_grads["sources"][param_name],
                    )
                else:
                    # For yoff and xoff, mask has correct shape already
                    masked_grads["sources"][param_name] = jnp.where(
                        zero_weight_mask,
                        jnp.zeros_like(masked_grads["sources"][param_name]),
                        masked_grads["sources"][param_name],
                    )

    # Beam parameters are NOT masked - they affect all sources
    return masked_grads


class BootstrapBeamFitter:
    """
    Bootstrap resampling over PolarizedBeamFitter with warm-starting.
    """

    def __init__(self, config):
        self.config = config
        self.base_fitter = PolarizedBeamFitter(config=self.config)
        self.original_fit_results = None
        self.ml_params_physical = None
        self.ml_params_logit = None

    def run_fit(self):
        """Run fitting with optional bootstrap uncertainty estimation."""
        print("=== Starting Polarized Beam Fitting ===")

        # Step 1 & 2: Original fit
        print("\n1. Running initial maximum likelihood fit...")
        self.original_fit_results = self.base_fitter.run_fit()

        # Store ML parameters for smart warm-starting
        self.ml_params_physical = self.base_fitter.params_physical
        self.ml_params_logit = self.base_fitter.params_logit

        # Add chi2 to original results
        self.original_fit_results["total_chi2"] = np.sum(self.base_fitter.latest_chi2s)

        # Initialize RNG key for bootstrap for reproducibility
        self.rng_key = jax.random.PRNGKey(self.config.bootstrap_seed)

        # Step 3 & 4: Prepare bootstrap weights and run iterations
        print(f"\n2. Preparing and running {self.config.n_bootstrap_samples} bootstrap iterations...")
        bootstrap_params = self._run_bootstrap_fits()

        # Step 5: Return results
        print(f"\nCompleted {len(bootstrap_params)} bootstrap samples")
        return {
            "ml_fit": self.original_fit_results,
            "bootstrap_fits": bootstrap_params,
        }

    def _prepare_bootstrap_weights(self):
        """Prepare bootstrap weight arrays."""
        n_sources = len(self.base_fitter.state.source_ids)

        # Generate all bootstrap indices at once
        self.rng_key, subkey = jax.random.split(self.rng_key)
        bootstrap_indices = jax.random.choice(
            subkey,
            n_sources,
            shape=(self.config.n_bootstrap_samples, n_sources),
            replace=True,
        )

        # Create weight arrays using vmap for efficiency
        def create_weight_array(indices):
            return jnp.bincount(indices, length=n_sources).astype(self.config.dtype_jax_real)

        return jax.vmap(create_weight_array)(bootstrap_indices)

    def _create_bootstrap_initial_params(self, weight_array):
        """
        Create smart initial parameters for a bootstrap sample based on source weights.
        This function is vmappable (no Python conditionals on traced values).

        Strategy:
        - Beam parameters: Always use ML solution (they affect all sources)
        - Source parameters:
          * Weight > 0: Use ML parameters (these sources are included)
          * Weight = 0: Use neutral/default values (these sources are excluded)
        """
        # Start with ML parameters as base
        initial_params_physical = jax.tree.map(lambda x: x, self.ml_params_physical)

        # Create masks for excluded sources (weight = 0)
        zero_weight_mask = weight_array == 0

        # Prepare neutral values for excluded sources
        neutral_yoff = jnp.zeros_like(initial_params_physical["sources"]["yoff"])
        neutral_xoff = jnp.zeros_like(initial_params_physical["sources"]["xoff"])
        neutral_flux = jnp.ones_like(initial_params_physical["sources"]["flux"])

        # Apply neutral values to excluded sources using jnp.where
        # This works regardless of whether any sources are excluded
        initial_params_physical["sources"]["yoff"] = jnp.where(zero_weight_mask, neutral_yoff, initial_params_physical["sources"]["yoff"])
        initial_params_physical["sources"]["xoff"] = jnp.where(zero_weight_mask, neutral_xoff, initial_params_physical["sources"]["xoff"])

        # For flux, expand mask to match flux array shape (n_sources, n_bands, 3)
        flux_shape = initial_params_physical["sources"]["flux"].shape
        zero_weight_mask_expanded = jnp.broadcast_to(zero_weight_mask.reshape(-1, 1, 1), flux_shape)
        initial_params_physical["sources"]["flux"] = jnp.where(
            zero_weight_mask_expanded, neutral_flux, initial_params_physical["sources"]["flux"]
        )

        # Convert to logit space for optimization
        return params_to_logit(initial_params_physical, self.config)

    def _build_bootstrap_objective(self):
        """
        Build a bootstrap objective function that takes data explicitly.
        This avoids closing over data in the JIT compilation.
        """
        # Create a fresh ObjectiveFunctions instance to build objectives
        obj_builder = ObjectiveFunctions(self.config, self.base_fitter.state, self.base_fitter.beam_models)

        # Build bootstrap-specific objective that includes weights
        if self.config.chi2_method == "fourier":

            def bootstrap_objective(params_logit, data, weight_array):
                """Bootstrap objective for Fourier space."""
                maps_fft, noise_psd = data
                params_phys = params_from_logit(params_logit, self.config)

                # Use the chi2 calculation from the objective builder
                vmap_chi2 = jax.vmap(obj_builder._chi2_fourier_single, in_axes=(None, 0, 0, 0, 0, 0))

                individual_chi2s = vmap_chi2(
                    params_phys["beams"],
                    params_phys["sources"]["yoff"],
                    params_phys["sources"]["xoff"],
                    params_phys["sources"]["flux"],
                    maps_fft,
                    noise_psd,
                )

                return jnp.sum(individual_chi2s * weight_array)
        else:

            def bootstrap_objective(params_logit, data, weight_array):
                """Bootstrap objective for real space."""
                maps, weights = data
                params_phys = params_from_logit(params_logit, self.config)

                # Use the chi2 calculation from the objective builder
                vmap_chi2 = jax.vmap(obj_builder._chi2_real_single, in_axes=(None, 0, 0, 0, 0, 0))

                individual_chi2s = vmap_chi2(
                    params_phys["beams"],
                    params_phys["sources"]["yoff"],
                    params_phys["sources"]["xoff"],
                    params_phys["sources"]["flux"],
                    maps,
                    weights,
                )

                return jnp.sum(individual_chi2s * weight_array)

        return bootstrap_objective

    def _run_bootstrap_fits(self):
        """Run all bootstrap fits with smart warm-starting and parallel execution."""
        bootstrap_weights = self._prepare_bootstrap_weights()

        # Build the bootstrap objective function once
        bootstrap_objective = self._build_bootstrap_objective()

        # Get data from base fitter
        objective_data = self.base_fitter.objective_data

        # Get available devices
        devices = jax.devices()
        n_devices = len(devices)
        print(f"Using {n_devices} device(s) for bootstrap fitting")

        # Process bootstrap samples in batches that match device count
        bootstrap_params = []
        n_samples = len(bootstrap_weights)

        # Process in batches of n_devices
        for batch_start in range(0, n_samples, n_devices):
            batch_end = min(batch_start + n_devices, n_samples)
            batch_weights = bootstrap_weights[batch_start:batch_end]
            batch_size = len(batch_weights)

            print(f"Processing bootstrap samples {batch_start + 1}-{batch_end}/{n_samples}")

            # If batch size < n_devices, pad with dummy weights
            if batch_size < n_devices:
                padding_size = n_devices - batch_size
                dummy_weights = jnp.zeros((padding_size,) + batch_weights.shape[1:])
                batch_weights = jnp.concatenate([batch_weights, dummy_weights])

            # Create initial parameters for this batch
            batch_initial_params = jax.vmap(self._create_bootstrap_initial_params)(batch_weights)

            # Replicate data across devices
            replicated_data = jax.tree.map(lambda x: jnp.array([x] * n_devices), objective_data)

            # Run optimization in parallel using pmap
            batch_results = self._run_batch_optimization_pmap(batch_initial_params, replicated_data, batch_weights, bootstrap_objective)

            # Extract only the actual results (not padding)
            for i in range(batch_size):
                bootstrap_params.append(batch_results[i])

        return bootstrap_params

    def _run_batch_optimization_pmap(self, batch_initial_params, replicated_data, batch_weights, bootstrap_objective):
        """Run a batch of bootstrap optimizations in parallel using pmap."""

        @jax.pmap
        def optimize_single(initial_params_logit, data, weight_array):
            """Optimize a single bootstrap sample on one device."""
            # Create optimizer
            optimizer = optax.adam(**self.config.adam_kwargs)
            opt_state = optimizer.init(initial_params_logit)
            current_params = initial_params_logit

            # Compile loss and gradient function for this objective
            loss_and_grad = jax.jit(jax.value_and_grad(lambda p: bootstrap_objective(p, data, weight_array)))

            # Run optimization
            for i in range(self.config.n_steps):
                loss, grads = loss_and_grad(current_params)

                # Mask gradients for excluded sources (weight = 0)
                masked_grads = mask_gradients_for_excluded_sources(grads, weight_array)

                # Check for convergence (simplified for pmap)
                optax.global_norm(masked_grads)
                # Note: We can't break early in pmap, so we just keep going

                updates, opt_state = optimizer.update(masked_grads, opt_state)
                current_params = optax.apply_updates(current_params, updates)

            # Convert back to physical parameters
            return params_from_logit(current_params, self.config)

        # Run parallel optimization
        return optimize_single(batch_initial_params, replicated_data, batch_weights)

    def calculate_individual_chi2s(self, params_phys):
        """Calculate chi2 for each source individually."""
        return self.base_fitter.calculate_individual_chi2s(params_phys)

    def create_model_maps(self, best_fit_params):
        return self.base_fitter.create_model_maps(best_fit_params)

    def create_beam_profile_maps(self, best_fit_params):
        return self.base_fitter.create_beam_profile_maps(best_fit_params)

    # Delegate properties that plotting system expects
    @property
    def bands(self):
        return self.base_fitter.config.bands

    @property
    def source_ids(self):
        return self.base_fitter.state.source_ids

    @property
    def latest_chi2s(self):
        return self.base_fitter.latest_chi2s

    @property
    def beam_models(self):
        return self.base_fitter.beam_models

    @property
    def noise_psd_numpy(self):
        return np.array(self.base_fitter.state.noise_psd_jax) if self.base_fitter.state.noise_psd_jax is not None else None

    @property
    def maps_numpy(self):
        return np.array(self.base_fitter.state.maps_jax)
