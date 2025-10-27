"""
Bootstrap resampling for polarized beam fitting.
 - Warm start
 - No compilation of functions that close over data
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .fitter import PolarizedBeamFitter
from .utils import check_convergence, init_convergence_state, params_from_logit, params_to_logit


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
                    zero_weight_mask_expanded = jnp.broadcast_to(zero_weight_mask[:, None, None], flux_grad_shape)
                    masked_grads["sources"][param_name] = jnp.where(
                        zero_weight_mask_expanded,
                        jnp.zeros_like(masked_grads["sources"][param_name]),
                        masked_grads["sources"][param_name],
                    )
                else:
                    # For yoff and xoff, mask has correct shape already
                    zero_weight_mask_yx = zero_weight_mask[:, None]
                    masked_grads["sources"][param_name] = jnp.where(
                        zero_weight_mask_yx,
                        jnp.zeros_like(masked_grads["sources"][param_name]),
                        masked_grads["sources"][param_name],
                    )

    # Beam parameters are NOT masked - they affect all sources
    return masked_grads


def build_bootstrap_chi2_fourier(config, beam_models, y_grid, x_grid, apod_mask_broadcast, idx_y=None, idx_x=None):
    """
    Build a chi2 function for Fourier space that doesn't close over data.

    Returns a function that takes (params, data, weight_array) as arguments.
    """

    def chi2_fourier_single(beam_params_list, yoff, xoff, flux, data_fft, precision):
        """Chi2 for single source in Fourier space."""
        # Build model for all bands
        maps_per_band = []
        for i, band in enumerate(config.bands):
            beam_model = beam_models[band]
            T_map, P_map = beam_model.evaluate_beam_maps(beam_params_list[i], yoff[i], xoff[i])
            maps_per_band.append((T_map, P_map))

        # Stack T and P maps for all bands
        T_stack = jnp.stack([m[0] for m in maps_per_band], axis=-1)
        P_stack = jnp.stack([m[1] for m in maps_per_band], axis=-1)

        # Create templates for T, Q, U
        templates = jnp.stack([T_stack, P_stack, P_stack], axis=-1)

        # Multiply by flux
        flux_reshaped = flux[None, None, :, :]
        model = templates * flux_reshaped

        # Apply apodization and FFT
        model_apod = model * apod_mask_broadcast
        model_fft = jnp.fft.fft2(model_apod, axes=(0, 1))
        if idx_y is not None and idx_x is not None:
            model_fft = jnp.take(model_fft, idx_y, axis=0)
            model_fft = jnp.take(model_fft, idx_x, axis=1)

        # Calculate chi2 with support for diagonal or full covariance weighting
        residual_fft = data_fft - model_fft
        if precision.ndim == 4:
            # chi2 = (jnp.abs(residual_fft)**2) * precision
            chi2 = jnp.real(residual_fft * jnp.conj(residual_fft)) * precision  # better derivatives?
            chi2_per_mode = jnp.sum(chi2, axis=(-2, -1))
            return jnp.mean(chi2_per_mode)

        # Multi-band precision matrices arrive per source with axes (ny, nx, n_bands, n_stokes, n_bands, n_stokes)
        ny, nx = residual_fft.shape[:2]
        n_bands = residual_fft.shape[-2]
        n_stokes = residual_fft.shape[-1]

        residual_vec = residual_fft.reshape(ny, nx, n_bands * n_stokes)
        precision_matrix = precision.reshape(ny, nx, n_bands * n_stokes, n_bands * n_stokes)
        weighted_vec = jnp.einsum("yxvw,yxw->yxv", precision_matrix, residual_vec)
        chi2_per_k = jnp.einsum("yxv,yxv->yx", jnp.conj(residual_vec), weighted_vec)
        return jnp.sum(jnp.real(chi2_per_k))

    def bootstrap_objective(params_logit, data, weight_array):
        """Bootstrap objective for Fourier space."""
        maps_fft, precision = data
        params_phys = params_from_logit(params_logit, config)

        # Use vmap for efficiency
        vmap_chi2 = jax.vmap(chi2_fourier_single, in_axes=(None, 0, 0, 0, 0, 0))

        individual_chi2s = vmap_chi2(
            params_phys["beams"],
            params_phys["sources"]["yoff"],
            params_phys["sources"]["xoff"],
            params_phys["sources"]["flux"],
            maps_fft,
            precision,
        )

        return jnp.sum(individual_chi2s * weight_array)

    return bootstrap_objective


def build_bootstrap_chi2_real(config, beam_models, y_grid, x_grid):
    """
    Build a chi2 function for real space that doesn't close over data.

    Returns a function that takes (params, data, weight_array) as arguments.
    """

    def chi2_real_single(beam_params_list, yoff, xoff, flux, data, weight):
        """Chi2 for single source in real space."""
        # Build model for all bands
        maps_per_band = []
        for i, band in enumerate(config.bands):
            beam_model = beam_models[band]
            T_map, P_map = beam_model.evaluate_beam_maps(beam_params_list[i], yoff[i], xoff[i])
            maps_per_band.append((T_map, P_map))

        # Stack T and P maps for all bands
        T_stack = jnp.stack([m[0] for m in maps_per_band], axis=-1)
        P_stack = jnp.stack([m[1] for m in maps_per_band], axis=-1)

        # Create templates for T, Q, U
        templates = jnp.stack([T_stack, P_stack, P_stack], axis=-1)

        # Multiply by flux
        flux_reshaped = flux[None, None, :, :]
        model = templates * flux_reshaped

        # Calculate chi2
        residual = data - model
        chi2 = jnp.einsum("...i,...ij,...j->...", residual, weight, residual)
        return jnp.sum(chi2)

    def bootstrap_objective(params_logit, data, weight_array):
        """Bootstrap objective for real space."""
        maps, weights = data
        params_phys = params_from_logit(params_logit, config)

        # Use vmap for efficiency
        vmap_chi2 = jax.vmap(chi2_real_single, in_axes=(None, 0, 0, 0, 0, 0))

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


class BootstrapBeamFitter:
    """
    Bootstrap wrapper for PolarizedBeamFitter
    """

    def __init__(self, config):
        self.config = config
        self.base_fitter = PolarizedBeamFitter(config=self.config)
        self.original_fit_results = None
        self.ml_params_physical = None
        self.ml_params_logit = None
        self.to_whitened = None
        self.from_whitened = None

        # Build and JIT the objective function once to avoid recompilation
        self.bootstrap_objective = self._build_bootstrap_objective()
        self._loss_and_grad = jax.jit(jax.value_and_grad(self.bootstrap_objective))

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
        n_sources = len(self.base_fitter.source_ids)

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

    def _create_bootstrap_initial_params_physical(self, weight_array):
        """
        Create smart initial parameters for a bootstrap sample based on source weights.

        Strategy:
        - Beam parameters: Always use ML solution (they affect all sources)
        - Source parameters:
          * Weight > 0: Use ML parameters (these sources are included)
          * Weight = 0: Use neutral/default values (these sources are excluded)

        Returns:
            Physical parameters (not logit-transformed)
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
        zero_weight_mask_yx = zero_weight_mask[:, None]
        initial_params_physical["sources"]["yoff"] = jnp.where(
            zero_weight_mask_yx, neutral_yoff, initial_params_physical["sources"]["yoff"]
        )
        initial_params_physical["sources"]["xoff"] = jnp.where(
            zero_weight_mask_yx, neutral_xoff, initial_params_physical["sources"]["xoff"]
        )

        # For flux, expand mask to match flux array shape (n_sources, n_bands, 3)
        flux_shape = initial_params_physical["sources"]["flux"].shape
        zero_weight_mask_expanded = jnp.broadcast_to(zero_weight_mask[:, None, None], flux_shape)
        initial_params_physical["sources"]["flux"] = jnp.where(
            zero_weight_mask_expanded, neutral_flux, initial_params_physical["sources"]["flux"]
        )

        return initial_params_physical

    def _build_bootstrap_objective(self):
        """
        Build a bootstrap objective function that takes data explicitly.
        This avoids closing over data in the JIT compilation.
        """
        # Build the appropriate objective based on chi2 method
        if self.config.chi2_method == "fourier":
            return build_bootstrap_chi2_fourier(
                self.config,
                self.base_fitter.beam_models,
                self.base_fitter.y_grid,
                self.base_fitter.x_grid,
                self.base_fitter.apod_mask_broadcast,
                self.base_fitter.idx_y,
                self.base_fitter.idx_x,
            )
        else:
            return build_bootstrap_chi2_real(self.config, self.base_fitter.beam_models, self.base_fitter.y_grid, self.base_fitter.x_grid)

    def _run_bootstrap_fits(self):
        """Run all bootstrap fits with smart warm-starting."""
        bootstrap_weights = self._prepare_bootstrap_weights()
        bootstrap_params = []

        # Get data from base fitter
        objective_data = self.base_fitter.objective_data

        for i, weight_array in enumerate(bootstrap_weights):
            print(f"Bootstrap iteration {i + 1}/{self.config.n_bootstrap_samples}")

            # Create smart initial parameters for this bootstrap sample
            initial_params_physical = self._create_bootstrap_initial_params_physical(weight_array)

            # Run tuned optimization with smart initialization
            optimized_params_physical = self._run_tuned_bootstrap(initial_params_physical, weight_array, objective_data)

            bootstrap_params.append(optimized_params_physical)

        return bootstrap_params

    def _run_tuned_bootstrap(self, initial_params_physical, weight_array, data):
        """
        Run the tuned two-stage optimizer for a single bootstrap iteration.
        """
        params_logit = params_to_logit(initial_params_physical, self.config)
        self.config.adam_variant = "adam"
        self.config.adam_kwargs = {"learning_rate": 1e-2}
        self.config.loss_history_length = 100
        self.config.n_steps = 8000
        params_logit = self._run_adam_bootstrap(params_logit, weight_array, data)
        self.config.adam_variant = "amsgrad"
        self.config.adam_kwargs = {"learning_rate": 1e-3}
        self.config.loss_history_length = 10
        self.config.n_steps = 1000
        params_logit = self._run_adam_bootstrap(params_logit, weight_array, data)
        return params_from_logit(params_logit, self.config)

    def _run_adam_bootstrap(self, initial_params_logit, weight_array, data):
        """
        Run Adam optimization for a single bootstrap iteration.
        Only optimizes parameters for sources with weight > 0.
        """
        optimizer = getattr(optax, self.config.adam_variant)(**self.config.adam_kwargs)
        opt_state = optimizer.init(initial_params_logit)
        current_params = initial_params_logit

        # Initialize convergence tracking
        convergence_state = init_convergence_state()

        # Calculate initial gradients, passing all required arguments to the JIT-compiled function
        loss, initial_grads = self._loss_and_grad(current_params, data, weight_array)
        initial_masked_grads = mask_gradients_for_excluded_sources(initial_grads, weight_array)
        initial_grad_norm = optax.global_norm(initial_masked_grads)

        for i in range(self.config.n_steps):
            # Pass all arguments to the JIT-compiled function
            loss, grads = self._loss_and_grad(current_params, data, weight_array)

            # Mask gradients for excluded sources (weight = 0)
            masked_grads = mask_gradients_for_excluded_sources(grads, weight_array)
            grad_norm = optax.global_norm(masked_grads)

            # check_convergence returns the updated state with the best parameters seen so far
            converged, message, convergence_state = check_convergence(
                loss, grad_norm, i, self.config, convergence_state, initial_grad_norm, current_params
            )

            if converged:
                print(f"  Converged at step {i}: {message}")
                if self.config.convergence_criterion == "loss_history":
                    print(f"  Best loss found: {convergence_state['best_loss']:.2f} at step {convergence_state['best_step']}")
                break

            updates, opt_state = optimizer.update(masked_grads, opt_state)
            current_params = optax.apply_updates(current_params, updates)

            if i % 100 == 0:
                print(f"  Step {i}: loss={loss:.2f}, |grad|={grad_norm:.3f}")

        # if convergence criterion was based on loss history:
        if convergence_state["best_params"] is not None:
            return convergence_state["best_params"]

        # fallback for other convergence criteria:
        return current_params

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
        return self.base_fitter.source_ids

    @property
    def latest_chi2s(self):
        return self.base_fitter.latest_chi2s

    @property
    def beam_models(self):
        return self.base_fitter.beam_models

    @property
    def precision_numpy(self):
        return np.array(self.base_fitter.precision_jax) if self.base_fitter.precision_jax is not None else None

    @property
    def maps_numpy(self):
        return np.array(self.base_fitter.maps_jax)
