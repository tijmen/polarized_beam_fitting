"""
Polarized beam fitter implementation.

Contains the PolarizedBeamFitter class that handles both ML optimization
and NUTS sampling, supports single and multi-band configurations, and provides
efficient parallelization across devices.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import blackjax
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import optax
import optimistix as optx
from numpyro.infer import MCMC, NUTS

from .beam_model import create_beam_model
from .cache import CacheManager
from .data_loader import DataLoader
from .noise_psd import create_noise_psd_calculator
from .utils import (
    build_whitening_transform,
    check_convergence,
    make_apodization_mask,
    params_from_logit,
    params_to_logit,
)


@dataclass
class FitterState:
    """Container for fitter state and precomputed data."""

    # Coordinate grids
    y_grid: jnp.ndarray
    x_grid: jnp.ndarray

    # Masks
    apod_mask: jnp.ndarray
    apod_mask_broadcast: jnp.ndarray

    # Data arrays
    maps_jax: jnp.ndarray
    weights_jax: Optional[jnp.ndarray] = None
    maps_fft_jax: Optional[jnp.ndarray] = None
    noise_psd_jax: Optional[jnp.ndarray] = None

    # Source information
    source_ids: np.ndarray = None
    n_src: int = 0
    n_bands: int = 0

    # Initial guesses
    gaussfit_yoff_numpy: np.ndarray = None
    gaussfit_xoff_numpy: np.ndarray = None
    gaussfit_initial_amp_numpy: np.ndarray = None
    gaussfit_initial_amp_jax: jnp.ndarray = None


class ObjectiveFunctions:
    """Encapsulates objective function logic."""

    def __init__(self, config, state: FitterState, beam_models: Dict):
        self.config = config
        self.state = state
        self.beam_models = beam_models

    def build_objective(self):
        """Build the appropriate objective function."""
        if self.config.chi2_method == "fourier":
            return self._build_fourier_objective()
        elif self.config.chi2_method == "real_space":
            return self._build_real_space_objective()
        else:
            raise ValueError(f"Unknown chi2_method: {self.config.chi2_method}")

    def _build_fourier_objective(self):
        """Build Fourier-space objective."""
        vmap_chi2 = jax.vmap(self._chi2_fourier_single, in_axes=(None, 0, 0, 0, 0, None))

        def objective(params_logit, data, extra_args=None):
            maps_fft, noise_psd = data
            params_phys = params_from_logit(params_logit, self.config)

            chi2_total = vmap_chi2(
                params_phys["beams"],
                params_phys["sources"]["yoff"],
                params_phys["sources"]["xoff"],
                params_phys["sources"]["flux"],
                maps_fft,
                noise_psd,
            ).sum()
            return chi2_total

        return objective

    def _build_real_space_objective(self):
        """Build real-space objective."""
        vmap_chi2 = jax.vmap(self._chi2_real_single, in_axes=(None, 0, 0, 0, 0, 0))

        def objective(params_logit, data, extra_args=None):
            maps, weights = data
            params_phys = params_from_logit(params_logit, self.config)

            chi2_total = vmap_chi2(
                params_phys["beams"],
                params_phys["sources"]["yoff"],
                params_phys["sources"]["xoff"],
                params_phys["sources"]["flux"],
                maps,
                weights,
            ).sum()
            return chi2_total

        return objective

    def _chi2_real_single(self, beam_params_list, yoff, xoff, flux, data, weight):
        """Chi2 for single source in real space."""
        model = self._build_model(beam_params_list, yoff, xoff, flux)
        residual = data - model
        chi2 = jnp.einsum("...i,...ij,...j->...", residual, weight, residual)
        return jnp.sum(chi2)

    def _chi2_fourier_single(self, beam_params_list, yoff, xoff, flux, data_fft, noise_psd):
        """Chi2 for single source in Fourier space."""
        model = self._build_model(beam_params_list, yoff, xoff, flux)
        model_apod = model * self.state.apod_mask_broadcast
        model_fft = jnp.fft.fft2(model_apod, axes=(0, 1))
        residual_fft = data_fft - model_fft

        # Full covariance not yet supported in this refactored version
        # Add 1e-30 to avoid division by zero
        chi2 = (jnp.abs(residual_fft) ** 2) / (noise_psd + 1e-30)
        return jnp.sum(jnp.mean(chi2, axis=(0, 1)))

    def _build_model(self, beam_params_list, yoff, xoff, flux):
        """Build beam model for all bands."""
        maps_per_band = []
        for i, band in enumerate(self.config.bands):
            beam_model = self.beam_models[band]
            T_map, P_map = beam_model.evaluate_beam_maps(beam_params_list[i], yoff, xoff)
            maps_per_band.append((T_map, P_map))

        # Stack T and P maps for all bands
        T_stack = jnp.stack([m[0] for m in maps_per_band], axis=-1)
        P_stack = jnp.stack([m[1] for m in maps_per_band], axis=-1)

        # Create templates for T, Q, U
        # Shape: (ny, nx, n_bands, 3)
        templates = jnp.stack([T_stack, P_stack, P_stack], axis=-1)

        # Multiply by flux: (n_bands, 3) -> (1, 1, n_bands, 3)
        flux_reshaped = flux[None, None, :, :]  # This line has been corrected from flux[:, None, None, :] to flux[None, None, :, :]

        # Final model shape: (ny, nx, n_bands, 3)
        model = templates * flux_reshaped
        return model


class PolarizedBeamFitter:
    """
    Refactored polarized beam fitting class.

    Cleaner interface and more modular design while maintaining
    compatibility with existing beam_model and noise_psd modules.
    """

    def __init__(self, config):
        """Initialize the fitter with configuration."""
        self.config = config
        self._setup_jax()
        self._print_welcome()

        # Initialize components
        self.state = self._initialize_state()
        self.beam_models = self._create_beam_models()

        # Load data
        self._load_data()

        # Build objective functions
        obj_builder = ObjectiveFunctions(self.config, self.state, self.beam_models)
        self.objective_function = obj_builder.build_objective()

        # Compile optimization functions
        self._loss_and_grad = jax.jit(jax.value_and_grad(lambda p, d: self.objective_function(p, d, None)))

        # Initialize parameters
        self.params_physical = self._initialize_parameters()
        self.params_logit = params_to_logit(self.params_physical, self.config)

    def _setup_jax(self):
        """Configure JAX settings."""
        if self.config.double_precision:
            jax.config.update("jax_enable_x64", True)
            print("64-bit precision enabled.")

    def _print_welcome(self):
        """Print welcome message."""
        print("\n" + "=" * 65)
        print("==        Welcome to the SPT-3G polarized beam fitter!        ==")
        print("== Questions? Contact Tijmen de Haan <tijmen.dehaan@gmail.com> ==")
        print("=" * 65 + "\n")
        print(f"Analysis for {self.config.bands}")

    def _initialize_state(self) -> FitterState:
        """Initialize fitter state with coordinate grids."""
        map_shape = (self.config.map_size_pix, self.config.map_size_pix)
        ny, nx = map_shape

        # Create coordinate grids
        y_coords = jnp.arange(-ny // 2, ny // 2, dtype=self.config.dtype_jax_real)
        x_coords = jnp.arange(-nx // 2, nx // 2, dtype=self.config.dtype_jax_real)
        y_grid = y_coords[:, None] * jnp.ones(nx, dtype=self.config.dtype_jax_real)
        x_grid = x_coords[None, :] * jnp.ones((ny, 1), dtype=self.config.dtype_jax_real)

        # Create apodization mask
        apod_mask = make_apodization_mask(map_shape, self.config.apodization_width_pix)
        apod_mask_jax = jnp.asarray(apod_mask, dtype=self.config.dtype_jax_real)

        return FitterState(
            y_grid=y_grid,
            x_grid=x_grid,
            apod_mask=apod_mask_jax,
            apod_mask_broadcast=apod_mask_jax[:, :, None, None],
            maps_jax=None,  # Will be filled during data loading
            n_bands=len(self.config.bands),
        )

    def _create_beam_models(self) -> Dict:
        """Create beam models for each band."""
        print("Creating beam models...")
        models = {}
        for band in self.config.bands:
            models[band] = create_beam_model(self.config, self.state.y_grid, self.state.x_grid, band)
        return models

    def _load_data(self):
        """Load and prepare data using cache if available."""
        print("Loading data...")

        # Use cache manager
        cache = CacheManager(self.config)
        loader = DataLoader(self.config)

        data = cache.load_or_create(loader.load_and_prepare)

        # Unpack data
        (
            gaussfit_yoff,
            gaussfit_xoff,
            gaussfit_amp,
            raw_maps,
            qu_templates,
            maps,
            weights,
            maps_fft,
            source_ids,
            n_src,
        ) = data

        # Store in state
        self.state.source_ids = source_ids
        self.state.n_src = n_src
        self.state.gaussfit_yoff_numpy = gaussfit_yoff
        self.state.gaussfit_xoff_numpy = gaussfit_xoff
        self.state.gaussfit_initial_amp_numpy = gaussfit_amp

        # Convert to JAX arrays
        self.state.gaussfit_initial_amp_jax = jnp.asarray(gaussfit_amp, dtype=self.config.dtype_jax_real)
        self.state.maps_jax = jnp.asarray(maps, dtype=self.config.dtype_jax_real)

        # Setup for specific chi2 method
        if self.config.chi2_method == "fourier":
            self._setup_fourier_data(maps, maps_fft)
        else:
            self._setup_real_space_data(weights)

    def _setup_fourier_data(self, maps, maps_fft):
        """Setup data for Fourier-space analysis."""
        self.state.maps_fft_jax = jnp.asarray(maps_fft, dtype=self.config.dtype_jax_complex)

        # Calculate noise PSD
        psd_calc = create_noise_psd_calculator(self.config, self.state.apod_mask.shape)
        noise_psd = psd_calc.calculate_noise_psd(maps)
        self.state.noise_psd_jax = jnp.asarray(noise_psd, dtype=self.config.dtype_jax_real)

        self.objective_data = (self.state.maps_fft_jax, self.state.noise_psd_jax)

    def _setup_real_space_data(self, weights):
        """Setup data for real-space analysis."""
        self.state.weights_jax = jnp.asarray(weights, dtype=self.config.dtype_jax_real)
        self.objective_data = (self.state.maps_jax, self.state.weights_jax)

    def _initialize_parameters(self) -> Dict:
        """Initialize fitting parameters."""
        params = {"beams": [], "sources": {}}

        # Initialize beam parameters
        for band in self.config.bands:
            beam_params = self.beam_models[band].get_initial_physical_params()
            params["beams"].append(
                jax.tree.map(
                    lambda x: jnp.asarray(x, dtype=self.config.dtype_jax_real),
                    beam_params,
                )
            )

        # Initialize source parameters
        yoff = self.state.gaussfit_yoff_numpy - 0.5
        xoff = self.state.gaussfit_xoff_numpy - 0.5

        params["sources"] = {
            "yoff": jnp.asarray(yoff, dtype=self.config.dtype_jax_real),
            "xoff": jnp.asarray(xoff, dtype=self.config.dtype_jax_real),
            "flux": self.state.gaussfit_initial_amp_jax,
        }

        return params

    def run_fit(self) -> Dict:
        """
        Run optimization to find best-fit parameters.

        Returns:
            Dictionary of best-fit physical parameters
        """
        print(f"Starting {self.config.solver} optimization...")

        if self.config.solver == "optimistix_bfgs":
            self._run_bfgs()
        elif self.config.solver == "optax_adam":
            self._run_adam()
        else:
            raise ValueError(f"Unknown solver: {self.config.solver}")

        # Calculate and report final chi2
        self._report_final_chi2()

        return self.params_physical

    def _run_bfgs(self):
        """Run BFGS optimization."""
        solver = optx.BFGS(**self.config.bfgs_kwargs)

        sol = optx.minimise(
            self.objective_function,
            solver,
            self.params_logit,
            args=self.objective_data,
            max_steps=self.config.n_steps,
            throw=False,
        )

        if sol.result != optx.RESULTS.successful:
            raise RuntimeError(f"BFGS failed: {optx.RESULTS[sol.result]}")

        self.params_logit = sol.value
        self.params_physical = params_from_logit(self.params_logit, self.config)

        print(f"Optimization finished after {sol.stats['num_steps']} steps.")

    def _run_adam(self):
        """Run Adam optimization."""
        optimizer = optax.adam(**self.config.adam_kwargs)
        opt_state = optimizer.init(self.params_logit)

        # Initialize convergence tracking
        convergence_state = {"loss_history": [], "best_loss": float("inf"), "best_step": -1}
        _, initial_grads = self._loss_and_grad(self.params_logit, self.objective_data)
        initial_grad_norm = optax.global_norm(initial_grads)

        for i in range(self.config.n_steps):
            loss, grads = self._loss_and_grad(self.params_logit, self.objective_data)
            grad_norm = optax.global_norm(grads)

            # Check convergence using the new unified criterion
            converged, message, best_loss = check_convergence(loss, grad_norm, i, self.config, convergence_state, initial_grad_norm)

            if converged:
                print(f"Converged at step {i}: {message}")
                if self.config.convergence_criterion == "loss_history":
                    print(f"Returning best loss found: {best_loss:.2f}")
                break

            updates, opt_state = optimizer.update(grads, opt_state)
            self.params_logit = optax.apply_updates(self.params_logit, updates)
            if i % 10 == 0:  # Print every 10 steps
                print(f"Step {i}/{self.config.n_steps}: loss={loss:.2f}, |grad|={grad_norm:.2f}")

        self.params_physical = params_from_logit(self.params_logit, self.config)

    def sample_with_mclmc(self) -> Dict:
        """
        Runs MCMC sampling using the dynamically-adjusted MCLMC algorithm.

        This implementation uses a pmapped manual warmup loop with dual-averaging
        step size adaptation for each chain.
        """
        devices = jax.devices()
        num_devices = len(devices)

        self._prepare_nuts_transform()

        def logdensity_fn(params_white, objective_data):
            """
            Computes the unnormalized log posterior density in the whitened space.

            The function includes a failsafe to return -inf for non-finite values
            of chi-squared, preventing numerical errors from halting the sampler.
            """
            params_phys = self.from_whitened(params_white)
            params_logit = params_to_logit(params_phys, self.config)
            chi2 = self.objective_function(params_logit, objective_data)

            # Failsafe for numerical instability
            log_prob = -0.5 * self.config.chi2_normalization * chi2 + self.log_det_jacobian
            return jnp.where(jnp.isfinite(chi2), log_prob, -jnp.inf)

        # Generate physically-jittered initial positions for each chain
        rng_key = jax.random.PRNGKey(0)
        print(f"Generating {num_devices} physically-jittered initial positions...")
        jittered_params_logit = self._make_jittered_inits(rng_key, num_devices)
        jittered_params_phys = jax.vmap(params_from_logit, in_axes=(0, None))(jittered_params_logit, self.config)
        init_positions = jax.vmap(self.to_whitened)(jittered_params_phys)

        # Replicate data for each device to be used in pmap
        replicated_data = jax.tree.map(lambda x: jnp.array([x] * num_devices), self.objective_data)

        @jax.pmap
        def run_chain(initial_position, rng_key, data):
            """
            Performs warmup and sampling for a single chain on a single device.
            """

            def sampler_logdensity(p):
                return logdensity_fn(p, data)

            # -- WARMUP: Dual-averaging adaptation for step size --
            warmup_rng, sample_rng = jax.random.split(rng_key)
            target_accept = 0.9
            da_init, da_update, _ = blackjax.dual_averaging.dual_averaging()
            default_step_size = 5.0
            adapt_state = da_init(default_step_size)  # Initial step size
            kernel = blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
                integration_steps_fn=lambda _: 10  # Fixed integration steps during warmup
            )

            def warmup_step(carry, rng_key):
                state, adapt_state = carry
                step_size = jnp.exp(adapt_state.log_x)

                next_state, info = kernel(rng_key, state, sampler_logdensity, step_size, L_proposal_factor=0.1)

                # Robustly calculate acceptance rate to update step size
                error_signal = target_accept - jnp.nan_to_num(info.acceptance_rate)
                new_adapt_state = da_update(adapt_state, error_signal)

                return (next_state, new_adapt_state), None

            initial_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(initial_position, sampler_logdensity, warmup_rng)

            (final_state, final_adapt_state), _ = jax.lax.scan(
                warmup_step, (initial_state, adapt_state), jax.random.split(warmup_rng, self.config.mcmc_num_warmup)
            )

            adapted_step_size = jnp.exp(final_adapt_state.log_x_avg)

            # Fall back to default step size if adapted value is outside reasonable bounds
            adapted_step_size = jnp.where((adapted_step_size < 0.1) | (adapted_step_size > 100.0), default_step_size, adapted_step_size)

            # -- SAMPLING --
            def sample_step(state, key):
                # just re-use the fixed integration step kernel, but alternatively could use e.g.
                #  dynamic_kernel = blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(integration_steps_fn=lambda key: jax.random.randint(key, (), 10, 25))
                next_state, info = kernel(key, state, sampler_logdensity, adapted_step_size, L_proposal_factor=0.1)
                return next_state, (next_state.position, info.acceptance_rate)

            _, (samples, acceptance_rates) = jax.lax.scan(
                sample_step, final_state, jax.random.split(sample_rng, self.config.mcmc_num_samples)
            )

            return samples, acceptance_rates, adapted_step_size

        # Distribute keys to devices and run the pmapped function
        keys = jax.random.split(rng_key, num_devices)
        print(f"Starting warmup and sampling with {num_devices} chains...")
        samples_pmap, acceptance_pmap, step_size_pmap = run_chain(init_positions, keys, replicated_data)

        mean_acceptance = jnp.mean(acceptance_pmap)
        mean_step_size = jnp.mean(step_size_pmap)
        print("\nSampling complete.")
        print(f"  Mean acceptance rate: {mean_acceptance:.3f}")
        print(f"  Mean adapted step size: {mean_step_size:.4f}")

        # Combine results from all chains
        samples_white = samples_pmap.reshape(-1, samples_pmap.shape[-1])
        samples_phys = jax.vmap(self.from_whitened)(samples_white)
        summary = self._calculate_summary_stats(samples_phys)

        return {
            "samples_white": samples_white,
            "samples_phys": samples_phys,
            "summary": summary,
            "adapted_params": {
                "step_size_per_chain": np.array(step_size_pmap),
                "mean_acceptance_rate": float(mean_acceptance),
            },
        }

    def _report_final_chi2(self):
        """Calculate and report final chi2 values."""
        chi2s = self.calculate_individual_chi2s(self.params_physical)
        self._latest_chi2s = np.array(chi2s)
        print(f"Final total chi2: {np.sum(self._latest_chi2s):.2f}")
        print(f"Mean chi2 per source: {np.mean(self._latest_chi2s):.2f}")

    def calculate_individual_chi2s(self, params_phys: Dict) -> jnp.ndarray:
        """Calculate chi2 for each source individually."""
        obj_builder = ObjectiveFunctions(self.config, self.state, self.beam_models)

        if self.config.chi2_method == "fourier":

            def chi2_fn(y, x, f, d):
                return obj_builder._chi2_fourier_single(params_phys["beams"], y, x, f, d, self.state.noise_psd_jax)

            chi2s = jax.vmap(chi2_fn, in_axes=(0, 0, 0, 0))(
                params_phys["sources"]["yoff"],
                params_phys["sources"]["xoff"],
                params_phys["sources"]["flux"],
                self.state.maps_fft_jax,
            )
        else:

            def chi2_fn(y, x, f, d, w):
                return obj_builder._chi2_real_single(params_phys["beams"], y, x, f, d, w)

            chi2s = jax.vmap(chi2_fn, in_axes=(0, 0, 0, 0, 0))(
                params_phys["sources"]["yoff"],
                params_phys["sources"]["xoff"],
                params_phys["sources"]["flux"],
                self.state.maps_jax,
                self.state.weights_jax,
            )

        return chi2s

    def _prepare_nuts_transform(self):
        """Calculate the Hessian at the MAP and build the whitening transform for NUTS."""
        print("Calculating Hessian at MAP for NUTS whitening transform...")

        # Define a function that takes physical parameters and returns the chi-squared value
        def objective_physical(params_phys):
            params_logit = params_to_logit(params_phys, self.config)
            return self.objective_function(params_logit, self.objective_data)

        # Efficiently calculate the diagonal of the Hessian (curvature)
        # This avoids instantiating the full Hessian, which is too large for memory.
        # See: https://github.com/google/jax/issues/3957
        flattened_params, unflatten_fn = jax.flatten_util.ravel_pytree(self.params_physical)

        def get_hessian_diag_element(i):
            # Define a function that takes the flattened parameter vector and returns the i-th element of the gradient
            grad_fn = jax.grad(lambda p: objective_physical(unflatten_fn(p)))
            # The i-th diagonal element of the Hessian is the i-th element of the gradient of the gradient
            return jax.grad(lambda p: grad_fn(p)[i])(flattened_params)[i]

        # Use lax.scan for a memory-efficient loop over parameters
        def body_fn(carry, i):
            return carry, get_hessian_diag_element(i)

        _, diag_hessian_flat = jax.lax.scan(body_fn, None, jnp.arange(len(flattened_params)))

        # Unflatten the diagonal Hessian back into a pytree
        curvature = unflatten_fn(diag_hessian_flat)

        # Build the transformation functions
        self.to_whitened, self.from_whitened, self.log_det_jacobian = build_whitening_transform(self.params_physical, curvature)
        print("Whitening transform for NUTS is ready.")

    def _make_jittered_inits(self, rng_key, num_chains: int):
        """
        Build per-chain initial points by adding small Gaussian noise to the
        current MAP parameters in *physical* space, then converting to logit.

        Jitter scales (1 σ):
            • beam parameters       : 0.001  (additive, absolute)
            • source y/x offsets    : 0.010  pixels
            • flux amplitudes       : 0.001  (relative, i.e. 1 %)
        """
        beam_phys = self.params_physical["beams"]
        src_phys = self.params_physical["sources"]

        def jitter_one_chain(key):
            key_beam, key_y, key_x, key_f = jax.random.split(key, 4)

            # --- beam params -------------------------------------------------- #
            def jitter_beam(bp, k):
                flat, treedef = jax.tree.flatten(bp)
                ks = jax.random.split(k, len(flat))
                jittered = [p + 0.0001 * jax.random.normal(kk, p.shape, p.dtype) for p, kk in zip(flat, ks)]
                return jax.tree.unflatten(treedef, jittered)

            beam_jittered = [jitter_beam(bp, k) for bp, k in zip(beam_phys, jax.random.split(key_beam, len(beam_phys)))]

            yoff_j = src_phys["yoff"] + 0.001 * jax.random.normal(key_y, src_phys["yoff"].shape)
            xoff_j = src_phys["xoff"] + 0.001 * jax.random.normal(key_x, src_phys["xoff"].shape)
            flux_j = src_phys["flux"] * (1.0 + 0.0001 * jax.random.normal(key_f, src_phys["flux"].shape))

            phys_jittered = {
                "beams": beam_jittered,
                "sources": {"yoff": yoff_j, "xoff": xoff_j, "flux": flux_j},
            }
            return params_to_logit(phys_jittered, self.config)

        chain_keys = jax.random.split(rng_key, num_chains)
        return jax.vmap(jitter_one_chain)(chain_keys)

    def sample_with_nuts(self) -> Dict:
        """
        Run NUTS sampling for uncertainty estimation.
        """
        num_chains = max(1, jax.local_device_count())

        # Prepare for NUTS sampling by calculating the Hessian for the whitening transform
        self._prepare_nuts_transform()

        print(f"Starting NUTS sampling: {num_chains} chains, {self.config.mcmc_num_warmup} warmup, {self.config.mcmc_num_samples} samples")

        # Define potential energy in the whitened space
        def potential_fn_whitened(params_white):
            # Transform from whitened space back to physical space
            params_phys = self.from_whitened(params_white)
            # The objective function expects logit params, so we do another conversion
            params_logit = params_to_logit(params_phys, self.config)
            chi2 = self.objective_function(params_logit, self.objective_data)
            # Return the potential energy, including the Jacobian correction term
            return 0.5 * self.config.chi2_normalization * chi2 - self.log_det_jacobian

        # Setup MCMC
        kernel = NUTS(
            potential_fn=potential_fn_whitened,
            target_accept_prob=self.config.mcmc_target_accept,
            max_tree_depth=self.config.mcmc_max_tree_depth,
        )

        mcmc = MCMC(
            kernel,
            num_warmup=self.config.mcmc_num_warmup,
            num_samples=self.config.mcmc_num_samples,
            num_chains=num_chains,
            chain_method="parallel" if num_chains > 1 else "sequential",
        )

        # Initialize and run in the whitened space
        # The initial points are draws from a standard normal distribution,
        # which corresponds to the whitened MAP.
        rng_key = jax.random.PRNGKey(0)
        map_white = self.to_whitened(self.params_physical)
        init_params_white = jax.random.normal(rng_key, (num_chains, map_white.shape[0]))

        if num_chains == 1:
            init_params_white = init_params_white[0]

        mcmc.run(rng_key, init_params=init_params_white, extra_fields=("potential_energy",))

        # Process results
        samples_white = mcmc.get_samples(group_by_chain=False)
        # Transform samples back to the physical space for analysis and plotting
        samples_phys = jax.vmap(self.from_whitened)(samples_white)

        # Calculate summary statistics
        summary = self._calculate_summary_stats(samples_phys)

        return {
            "samples_white": samples_white,
            "samples_phys": samples_phys,
            "summary": summary,
            "mcmc": mcmc,
        }

    def _calculate_summary_stats(self, samples: Dict) -> Dict:
        """Calculate summary statistics from samples."""

        def get_stats(x):
            return {
                "mean": jnp.mean(x, axis=0),
                "std": jnp.std(x, axis=0),
                "q16": jnp.percentile(x, 16, axis=0),
                "q50": jnp.percentile(x, 50, axis=0),
                "q84": jnp.percentile(x, 84, axis=0),
            }

        return jax.tree.map(get_stats, samples)

    def create_model_maps(self, params: Dict) -> np.ndarray:
        """
        Create model maps for all sources.

        Args:
            params: Physical parameters dictionary

        Returns:
            Array of model maps with shape (n_src, ny, nx, n_bands, 3)
        """
        obj_builder = ObjectiveFunctions(self.config, self.state, self.beam_models)

        def model_for_source(yoff, xoff, flux):
            return obj_builder._build_model(params["beams"], yoff, xoff, flux)

        # Vectorize over sources
        models = jax.vmap(model_for_source)(params["sources"]["yoff"], params["sources"]["xoff"], params["sources"]["flux"])

        return np.array(models)

    def create_beam_profile_maps(self, params: Dict) -> Tuple[Dict, Dict]:
        """
        Create centered beam maps for profile analysis.

        Args:
            params: Physical parameters dictionary

        Returns:
            Tuple of (T_maps, P_maps) dictionaries keyed by band
        """
        T_maps = {}
        P_maps = {}

        for i, band in enumerate(self.config.bands):
            T_map, P_map = self.beam_models[band].evaluate_beam_maps(params["beams"][i], 0.0, 0.0)
            T_maps[band] = np.array(T_map)
            P_maps[band] = np.array(P_map)

        return T_maps, P_maps

    def _get_latest_chi2s(self):
        """Get latest chi2 values."""
        if hasattr(self, "_latest_chi2s"):
            return self._latest_chi2s
        else:
            # Calculate if not available
            return np.array(self.calculate_individual_chi2s(self.params_physical))

    @property
    def latest_chi2s(self):
        """Access latest chi2 values."""
        return self._get_latest_chi2s()
