"""
Bootstrap uncertainty estimation for polarized beam fitting.

Improved implementation with smart warm-starting based on bootstrap weights.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .fitter import PolarizedBeamFitter
from .utils import check_convergence, params_from_logit, params_to_logit


def make_bootstrap_objective(get_individual_chi2s_func, config):
    """
    Creates and JIT-compiles a bootstrap objective function that takes weight_array as data.
    """

    @jax.jit
    def bootstrap_objective(params_logit, weight_array):
        # Convert logit parameters to physical space for chi2 calculation
        params_phys = params_from_logit(params_logit, config)
        individual_chi2s = get_individual_chi2s_func(params_phys)
        return jnp.sum(individual_chi2s * weight_array)

    return bootstrap_objective


def mask_gradients_for_excluded_sources(grad_tree, weight_array):
    """
    Zero out gradients for source parameters where weight = 0.
    Beam parameters are always kept since they affect all sources.
    """
    zero_weight_mask = weight_array == 0

    # Create a copy of the gradient tree
    masked_grads = jax.tree_util.tree_map(lambda x: x, grad_tree)

    # Mask source parameter gradients for excluded sources
    if "sources" in masked_grads:
        for param_name in ["yoff", "xoff", "flux"]:
            if param_name in masked_grads["sources"]:
                if param_name == "flux":
                    # For flux gradients, need to expand mask to match shape (n_sources, n_bands, 3)
                    flux_grad_shape = masked_grads["sources"][param_name].shape
                    zero_weight_mask_expanded = jnp.broadcast_to(zero_weight_mask.reshape(-1, 1, 1), flux_grad_shape)
                    masked_grads["sources"][param_name] = jnp.where(
                        zero_weight_mask_expanded, jnp.zeros_like(masked_grads["sources"][param_name]), masked_grads["sources"][param_name]
                    )
                else:
                    # For yoff and xoff, mask has correct shape already
                    masked_grads["sources"][param_name] = jnp.where(
                        zero_weight_mask, jnp.zeros_like(masked_grads["sources"][param_name]), masked_grads["sources"][param_name]
                    )

    # Beam parameters are NOT masked - they affect all sources
    return masked_grads


class BootstrapBeamFitter:
    """
    Bootstrap wrapper for PolarizedBeamFitter with smart warm-starting.

    Improved implementation that:
    1. Runs initial ML fit to get baseline parameters
    2. For each bootstrap sample, intelligently initializes parameters based on
       which sources are included (weight > 0) vs excluded (weight = 0)
    3. Beam parameters always start from ML solution (affect all sources)
    4. Source parameters are initialized based on bootstrap weights:
       - Weight > 0: Use ML parameters as starting point
       - Weight = 0: Use neutral/default values (won't affect optimization)
    5. Tracks convergence and provides detailed diagnostics
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

        if not self.config.enable_bootstrap:
            print("Bootstrap disabled. Returning ML results.")
            return self.original_fit_results

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

        Strategy:
        - Beam parameters: Always use ML solution (they affect all sources)
        - Source parameters:
          * Weight > 0: Use ML parameters (these sources are included)
          * Weight = 0: Use neutral/default values (these sources are excluded)

        Args:
            weight_array: Bootstrap weights for sources [n_sources]

        Returns:
            Initial parameters in logit space for this bootstrap sample
        """
        # Start with ML parameters as base
        initial_params_physical = jax.tree_util.tree_map(lambda x: x, self.ml_params_physical)

        # For sources with weight = 0, set to neutral values
        zero_weight_mask = weight_array == 0
        n_zero_sources = jnp.sum(zero_weight_mask)

        if n_zero_sources > 0:
            # Set neutral values for excluded sources
            # These values won't affect the objective, but should be reasonable for optimization

            # Small offset values (near zero but not exactly zero to avoid numerical issues)
            neutral_yoff = jnp.zeros_like(initial_params_physical["sources"]["yoff"])
            neutral_xoff = jnp.zeros_like(initial_params_physical["sources"]["xoff"])

            # Unit flux values (neutral multiplicative factor)
            neutral_flux = jnp.ones_like(initial_params_physical["sources"]["flux"])

            # Apply neutral values only to excluded sources
            initial_params_physical["sources"]["yoff"] = jnp.where(
                zero_weight_mask, neutral_yoff, initial_params_physical["sources"]["yoff"]
            )
            initial_params_physical["sources"]["xoff"] = jnp.where(
                zero_weight_mask, neutral_xoff, initial_params_physical["sources"]["xoff"]
            )

            # For flux, need to expand mask to match flux array shape (n_sources, n_bands, 3)
            flux_shape = initial_params_physical["sources"]["flux"].shape
            zero_weight_mask_expanded = jnp.broadcast_to(zero_weight_mask.reshape(-1, 1, 1), flux_shape)
            initial_params_physical["sources"]["flux"] = jnp.where(
                zero_weight_mask_expanded, neutral_flux, initial_params_physical["sources"]["flux"]
            )

        # Convert to logit space for optimization
        return params_to_logit(initial_params_physical, self.config)

    def _run_bootstrap_fits(self):
        """Run all bootstrap fits with smart warm-starting."""
        bootstrap_weights = self._prepare_bootstrap_weights()
        bootstrap_params = []

        for i, weight_array in enumerate(bootstrap_weights):
            print(f"Bootstrap iteration {i + 1}/{self.config.n_bootstrap_samples}")

            # Create smart initial parameters for this bootstrap sample
            initial_params_logit = self._create_bootstrap_initial_params(weight_array)

            # Build a dedicated objective function for this bootstrap sample
            objective_func = make_bootstrap_objective(self.base_fitter.calculate_individual_chi2s, self.config)

            # Run Adam optimization with smart initialization and gradient masking
            optimized_params_logit = self._run_adam_bootstrap(objective_func, initial_params_logit, weight_array)

            # Convert logit parameters back to physical space for storage
            physical_params = params_from_logit(optimized_params_logit, self.config)
            bootstrap_params.append(physical_params)

        return bootstrap_params

    def _run_adam_bootstrap(self, objective_func, initial_params_logit, weight_array):
        """
        Run Adam optimization for a single bootstrap iteration.
        Only optimizes parameters for sources with weight > 0.
        """
        optimizer = optax.adam(**self.config.adam_kwargs)
        opt_state = optimizer.init(initial_params_logit)
        current_params = initial_params_logit

        # Compile loss and gradient function for this objective
        loss_and_grad = jax.jit(jax.value_and_grad(objective_func))

        # Initialize convergence tracking
        convergence_state = {"loss_history": [], "best_loss": float("inf"), "best_step": -1}
        _, initial_grads = loss_and_grad(current_params, weight_array)
        initial_masked_grads = mask_gradients_for_excluded_sources(initial_grads, weight_array)
        initial_grad_norm = optax.global_norm(initial_masked_grads)

        for i in range(self.config.n_steps):
            loss, grads = loss_and_grad(current_params, weight_array)

            # Mask gradients for excluded sources (weight = 0)
            masked_grads = mask_gradients_for_excluded_sources(grads, weight_array)

            grad_norm = optax.global_norm(masked_grads)

            # Check convergence using the new unified criterion
            converged, message, best_loss = check_convergence(loss, grad_norm, i, self.config, convergence_state, initial_grad_norm)

            if converged:
                print(f"  Converged at step {i}: {message}")
                if self.config.convergence_criterion == "loss_history":
                    print(f"  Returning best loss found: {best_loss:.2f}")
                break

            updates, opt_state = optimizer.update(masked_grads, opt_state)
            current_params = optax.apply_updates(current_params, updates)

            if i % 100 == 0:
                print(f"  Step {i}: loss={loss:.2f}, |grad|={grad_norm:.3f}")

        return current_params

    # Delegate methods to base fitter

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
