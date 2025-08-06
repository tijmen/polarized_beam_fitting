"""
Polarized beam fitter implementation.

Contains the PolarizedBeamFitter class that handles both ML optimization
and NUTS sampling, supports single and multi-band configurations, and provides
efficient parallelization across devices.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
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

        for i in range(self.config.n_steps):
            loss, grads = self._loss_and_grad(self.params_logit, self.objective_data)
            updates, opt_state = optimizer.update(grads, opt_state)
            self.params_logit = optax.apply_updates(self.params_logit, updates)

            grad_norm = optax.global_norm(grads)

            if i % 10 == 0:  # Print every 10 steps
                print(f"Step {i}/{self.config.n_steps}: loss={loss:.2f}, |grad|={grad_norm:.2f}")

            if grad_norm < 1:
                print(f"Converged at step {i}")
                break

        self.params_physical = params_from_logit(self.params_logit, self.config)

    def _report_final_chi2(self):
        """Calculate and report final chi2 values."""
        chi2s = self.calculate_individual_chi2s(self.params_physical)
        self.latest_chi2s = np.array(chi2s)
        print(f"Final total chi2: {np.sum(self.latest_chi2s):.2f}")
        print(f"Mean chi2 per source: {np.mean(self.latest_chi2s):.2f}")

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

    def sample_with_nuts(self, num_warmup: int = None, num_samples: int = None) -> Dict:
        """
        Run NUTS sampling for uncertainty estimation.

        Args:
            num_warmup: Number of warmup steps (default from config)
            num_samples: Number of samples (default from config)

        Returns:
            Dictionary with samples and summary statistics
        """
        num_warmup = num_warmup or self.config.nuts_num_warmup
        num_samples = num_samples or self.config.nuts_num_samples
        num_chains = max(1, jax.local_device_count())

        print(f"Starting NUTS sampling: {num_chains} chains, {num_warmup} warmup, {num_samples} samples")

        # Define potential energy
        def potential_fn(params_logit):
            chi2, _ = self._loss_and_grad(params_logit, self.objective_data)
            return 0.5 * self.config.chi2_normalization * chi2

        # Setup MCMC
        kernel = NUTS(
            potential_fn=potential_fn,
            target_accept_prob=self.config.nuts_target_accept,
            max_tree_depth=self.config.nuts_max_tree_depth,
        )

        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method="parallel" if num_chains > 1 else "sequential",
        )

        # Initialize and run
        rng_key = jax.random.PRNGKey(0)
        if num_chains > 1:
            rng_key = jax.random.split(rng_key, num_chains)
            init_params = jax.tree.map(
                lambda x: jnp.broadcast_to(x, (num_chains,) + x.shape),
                self.params_logit,
            )
        else:
            init_params = self.params_logit

        mcmc.run(rng_key, init_params=init_params, extra_fields=("potential_energy",))

        # Process results
        samples_logit = mcmc.get_samples(group_by_chain=False)
        samples_phys = jax.vmap(lambda p: params_from_logit(p, self.config))(samples_logit)

        # Calculate summary statistics
        summary = self._calculate_summary_stats(samples_phys)

        return {
            "samples_logit": samples_logit,
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
