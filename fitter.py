"""
Polarized beam fitter implementation.

Contains the PolarizedBeamFitter class that handles both ML optimization
and NUTS sampling, supports single and multi-band configurations, and provides
efficient parallelization across devices.
"""

import time
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
    calculate_tod_nyquist_radial_mask_smooth,
    check_convergence,
    compute_rectangular_ell_cut_indices,
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
    precision_jax: Optional[jnp.ndarray] = None
    k_mask_jax: Optional[jnp.ndarray] = None
    k_indices_y: Optional[jnp.ndarray] = None
    k_indices_x: Optional[jnp.ndarray] = None

    # Source information
    source_ids: np.ndarray = None
    n_src: int = 0
    fields: np.ndarray = None
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
        vmap_chi2 = jax.vmap(self._chi2_fourier_single, in_axes=(None, 0, 0, 0, 0, 0))

        def objective(params_logit, data, extra_args=None):
            maps_fft, precision = data
            params_phys = params_from_logit(params_logit, self.config)

            chi2_total = vmap_chi2(
                params_phys["beams"],
                params_phys["sources"]["yoff"],
                params_phys["sources"]["xoff"],
                params_phys["sources"]["flux"],
                maps_fft,
                precision,
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

    def _chi2_fourier_single(self, beam_params_list, yoff, xoff, flux, data_fft, precision):
        """Chi2 for single source in Fourier space."""
        model = self._build_model(beam_params_list, yoff, xoff, flux)
        model_apod = model * self.state.apod_mask_broadcast
        model_fft = jnp.fft.fft2(model_apod, axes=(0, 1))
        if self.state.k_indices_y is not None and self.state.k_indices_x is not None:
            model_fft = jnp.take(model_fft, self.state.k_indices_y, axis=0)
            model_fft = jnp.take(model_fft, self.state.k_indices_x, axis=1)
        residual_fft = data_fft - model_fft
        if precision.ndim == 4:
            chi2 = (jnp.abs(residual_fft) ** 2) * precision
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
        return jnp.mean(jnp.real(chi2_per_k))

    def _build_model(self, beam_params_list, yoff, xoff, flux):
        """Build beam model for all bands."""
        maps_per_band = []
        for i, band in enumerate(self.config.bands):
            beam_model = self.beam_models[band]
            T_map, P_map = beam_model.evaluate_beam_maps(beam_params_list[i], yoff[i], xoff[i])
            maps_per_band.append((T_map, P_map))

        # Stack T and P maps for all bands
        T_stack = jnp.stack([m[0] for m in maps_per_band], axis=-1)
        P_stack = jnp.stack([m[1] for m in maps_per_band], axis=-1)

        # Create templates for T, Q, U
        # Shape: (ny, nx, n_bands, 3)
        templates = jnp.stack([T_stack, P_stack, P_stack], axis=-1)

        # Multiply by flux: (n_bands, 3) -> (1, 1, n_bands, 3)
        flux_reshaped = flux[None, None, :, :]

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
        self._loss_and_grad = jax.jit(jax.value_and_grad(lambda params_logit, data: self.objective_function(params_logit, data, None)))

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
        apod_width = self.config.apodization_width_pix
        ny, nx = map_shape

        # Create coordinate grids
        y_coords = jnp.arange(-ny // 2, ny // 2, dtype=self.config.dtype_jax_real)
        x_coords = jnp.arange(-nx // 2, nx // 2, dtype=self.config.dtype_jax_real)
        y_grid = y_coords[:, None] * jnp.ones(nx, dtype=self.config.dtype_jax_real)
        x_grid = x_coords[None, :] * jnp.ones((ny, 1), dtype=self.config.dtype_jax_real)

        # Create apodization mask
        apod_mask = make_apodization_mask(map_shape, apod_width)
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
            source_fields,
            n_src,
        ) = data

        # Store in state
        self.state.source_ids = source_ids
        self.state.n_src = n_src
        self.state.gaussfit_yoff_numpy = gaussfit_yoff
        self.state.gaussfit_xoff_numpy = gaussfit_xoff
        self.state.gaussfit_initial_amp_numpy = gaussfit_amp
        self.state.fields = source_fields

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
        ny, nx = self.state.apod_mask.shape

        idx_y_np, idx_x_np = self._compute_ell_cut_indices((ny, nx))

        if idx_y_np is None or idx_x_np is None:
            self.state.k_indices_y = None
            self.state.k_indices_x = None
            maps_fft_cut = maps_fft
            ny_fft, nx_fft = ny, nx
        else:
            self.state.k_indices_y = jnp.asarray(idx_y_np, dtype=jnp.int32)
            self.state.k_indices_x = jnp.asarray(idx_x_np, dtype=jnp.int32)
            maps_fft_cut = self._truncate_fourier_numpy(maps_fft, idx_y_np, idx_x_np, axis_y=1, axis_x=2)
            ny_fft, nx_fft = maps_fft_cut.shape[1:3]

        self.state.maps_fft_jax = jnp.asarray(maps_fft_cut, dtype=self.config.dtype_jax_complex)

        psd_calc = create_noise_psd_calculator(self.config, self.state.apod_mask.shape)
        self.state.k_mask_jax = None

        if self.config.noise_psd_method == "multiband_covariance":
            n_src = self.state.n_src
            n_bands = len(self.config.bands)
            n_stokes = 3
            precision_per_source = np.zeros(
                (n_src, ny_fft, nx_fft, n_bands, n_stokes, n_bands, n_stokes), dtype=self.config.dtype_np_complex
            )

            unique_fields = np.unique(self.state.fields)
            for field in unique_fields:
                field_indices = np.where(self.state.fields == field)[0]
                field_maps = maps[field_indices]
                covariance_psd = psd_calc.calculate_noise_psd(field_maps)
                precision_matrix = self._prepare_precision_from_covariance(covariance_psd)
                if idx_y_np is not None and idx_x_np is not None:
                    precision_matrix = self._truncate_fourier_numpy(precision_matrix, idx_y_np, idx_x_np, axis_y=0, axis_x=1)
                repeated_precision = np.broadcast_to(precision_matrix, (len(field_indices),) + precision_matrix.shape)
                precision_per_source[field_indices] = repeated_precision

            self.state.precision_jax = jnp.asarray(precision_per_source, dtype=self.config.dtype_jax_complex)
        elif self.config.noise_psd_method in {"pca_multiband_covariance", "parametric_prefit"}:
            n_src = self.state.n_src
            n_bands = len(self.config.bands)
            n_stokes = 3
            precision_per_source = np.zeros(
                (n_src, ny_fft, nx_fft, n_bands, n_stokes, n_bands, n_stokes), dtype=self.config.dtype_np_complex
            )

            unique_fields = np.unique(self.state.fields)
            for field in unique_fields:
                field_indices = np.where(self.state.fields == field)[0]
                field_maps = maps[field_indices]
                precision_matrix = psd_calc.calculate_noise_psd(field_maps)
                if idx_y_np is not None and idx_x_np is not None:
                    precision_matrix = self._truncate_fourier_numpy(precision_matrix, idx_y_np, idx_x_np, axis_y=0, axis_x=1)
                repeated_precision = np.broadcast_to(precision_matrix, (len(field_indices),) + precision_matrix.shape)
                precision_per_source[field_indices] = repeated_precision

            self.state.precision_jax = jnp.asarray(precision_per_source, dtype=self.config.dtype_jax_complex)
        else:
            original_noise_psd = psd_calc.calculate_noise_psd(maps)
            with np.errstate(divide="ignore", invalid="ignore"):
                precision_diag = np.reciprocal(original_noise_psd)
            precision_diag = np.where(np.isfinite(precision_diag), precision_diag, 0.0)
            if precision_diag.ndim == 4:
                precision_diag = np.broadcast_to(precision_diag, (self.state.n_src,) + precision_diag.shape)
            if idx_y_np is not None and idx_x_np is not None:
                if precision_diag.ndim == 4:
                    precision_diag = self._truncate_fourier_numpy(precision_diag, idx_y_np, idx_x_np, axis_y=0, axis_x=1)
                elif precision_diag.ndim == 5:
                    precision_diag = self._truncate_fourier_numpy(precision_diag, idx_y_np, idx_x_np, axis_y=1, axis_x=2)
                elif precision_diag.ndim == 6:
                    precision_diag = self._truncate_fourier_numpy(precision_diag, idx_y_np, idx_x_np, axis_y=0, axis_x=1)
                elif precision_diag.ndim == 7:
                    precision_diag = self._truncate_fourier_numpy(precision_diag, idx_y_np, idx_x_np, axis_y=1, axis_x=2)
                else:
                    raise ValueError(f"Unexpected precision tensor rank: {precision_diag.ndim}")
            dtype = self.config.dtype_jax_complex if np.iscomplexobj(precision_diag) else self.config.dtype_jax_real
            self.state.precision_jax = jnp.asarray(precision_diag, dtype=dtype)

        self._apply_fourier_mask()
        self.objective_data = (self.state.maps_fft_jax, self.state.precision_jax)

    def _setup_real_space_data(self, weights):
        """Setup data for real-space analysis."""
        self.state.weights_jax = jnp.asarray(weights, dtype=self.config.dtype_jax_real)
        self.objective_data = (self.state.maps_jax, self.state.weights_jax)

    def _calculate_nyquist_mask(self, source_ids):
        """Return radial Fourier mask per source (values in [0, 1])."""
        map_shape = tuple(int(dim) for dim in self.state.apod_mask.shape)
        masks = [calculate_tod_nyquist_radial_mask_smooth(sid, map_shape, self.config) for sid in source_ids]
        return np.stack(masks)  # (n_src, ny, nx)

    def _compute_ell_cut_indices(self, map_shape):
        """Return ky/kx indices obeying ell <= ellmax; None if no cutoff requested."""
        ellmax = getattr(self.config, "ellmax", None)
        return compute_rectangular_ell_cut_indices(map_shape, self.config.reso_arcmin, ellmax)

    def _truncate_fourier_numpy(self, array, idx_y, idx_x, axis_y, axis_x):
        """Return array truncated along specified Fourier axes if indices are provided."""
        if idx_y is None or idx_x is None:
            return array
        truncated = np.take(array, idx_y, axis=axis_y)
        truncated = np.take(truncated, idx_x, axis=axis_x)
        return truncated

    def _apply_fourier_mask(self):
        """Multiply precision weights by the source-specific Fourier mask."""
        if self.state.k_mask_jax is None or self.state.precision_jax is None:
            return

        mask = self.state.k_mask_jax
        precision = self.state.precision_jax

        if precision.ndim == 5:  # (n_src, ny, nx, n_bands, 3)
            mask_broadcast = mask[..., None, None]
        elif precision.ndim == 7:  # (n_src, ny, nx, n_bands, 3, n_bands, 3)
            mask_broadcast = mask[..., None, None, None, None]
        else:
            raise ValueError(f"Unexpected precision tensor rank: {precision.ndim}")

        self.state.precision_jax = precision * mask_broadcast

    def _prepare_precision_from_covariance(self, covariance_psd):
        """Invert covariance matrices for multi-band chi^2 weighting."""
        print("Preparing inverse covariance matrices for multi-band chi^2...")

        cov = np.asarray(covariance_psd, dtype=self.config.dtype_np_complex)

        if cov.ndim != 6:
            raise ValueError(f"Expected 6D covariance tensor, received shape {cov.shape}")

        ny, nx, n_bands, n_stokes, cov_band, cov_stokes = cov.shape
        if cov_band != n_bands or cov_stokes != n_stokes:
            raise ValueError(f"Covariance tensor axes must follow (n_bands, n_stokes, n_bands, n_stokes); received shape {cov.shape}")

        n_dim = n_bands * n_stokes

        cov_matrix = cov.reshape(ny, nx, n_dim, n_dim)
        cov_matrix = 0.5 * (cov_matrix + np.swapaxes(cov_matrix, -1, -2).conj())

        reg = getattr(self.config, "multiband_covariance_regularization", 1e-3)
        eye = np.eye(n_dim, dtype=self.config.dtype_np_real)
        cov_matrix += reg * eye[None, None, :, :]

        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                "Failed to invert multi-band covariance matrices; consider increasing multiband_covariance_regularization."
            ) from exc

        inv_cov = inv_cov.reshape(ny, nx, n_bands, n_stokes, n_bands, n_stokes)
        inv_cov = 0.5 * (inv_cov + np.swapaxes(inv_cov, 2, 4).swapaxes(3, 5).conj())
        return inv_cov.astype(self.config.dtype_np_complex)

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

        yoff_per_band = np.repeat(yoff[:, None], self.state.n_bands, axis=1)
        xoff_per_band = np.repeat(xoff[:, None], self.state.n_bands, axis=1)

        params["sources"] = {
            "yoff": jnp.asarray(yoff_per_band, dtype=self.config.dtype_jax_real),
            "xoff": jnp.asarray(xoff_per_band, dtype=self.config.dtype_jax_real),
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
        convergence_state = {"loss_history": [], "best_loss": float("inf"), "best_step": -1, "best_params": None}
        _, initial_grads = self._loss_and_grad(self.params_logit, self.objective_data)
        initial_grad_norm = optax.global_norm(initial_grads)

        for i in range(self.config.n_steps):
            loss, grads = self._loss_and_grad(self.params_logit, self.objective_data)
            grad_norm = optax.global_norm(grads)

            # Check convergence and track best parameters
            converged, message, best_loss, best_params = check_convergence(
                loss, grad_norm, i, self.config, convergence_state, initial_grad_norm, self.params_logit
            )

            if converged:
                print(f"Converged at step {i}: {message}")
                if self.config.convergence_criterion == "loss_history":
                    print(f"Returning best loss found: {best_loss:.2f}")
                break

            updates, opt_state = optimizer.update(grads, opt_state)
            self.params_logit = optax.apply_updates(self.params_logit, updates)
            if i % 10 == 0:  # Print every 10 steps
                print(f"Step {i}/{self.config.n_steps}: loss={loss:.2f}, |grad|={grad_norm:.2f}")

        # Use best parameters if available, otherwise use current parameters
        if convergence_state["best_params"] is not None:
            self.params_logit = convergence_state["best_params"]
        self.params_physical = params_from_logit(self.params_logit, self.config)

    def sample_with_mclmc(self):
        """
        Single-chain MCLMC
        """
        self._prepare_nuts_transform()
        chi2_norm = jnp.asarray(self.config.chi2_normalization, dtype=self.config.dtype_jax_real)
        log_det = jnp.asarray(self.log_det_jacobian, dtype=self.config.dtype_jax_real)

        def logdensity_fn(params_white):
            params_phys = self.from_whitened(params_white)
            params_logit = params_to_logit(params_phys, self.config)
            chi2 = self.objective_function(params_logit, self.objective_data)
            return -0.5 * chi2_norm * chi2 + log_det

        rng_key = jax.random.PRNGKey(42)
        init_key, tune_key, run_key = jax.random.split(rng_key, 3)

        print("Generating physically-jittered initial position...")
        init_params_logit = self.params_logit
        init_params_phys = params_from_logit(init_params_logit, self.config)
        init_white = self.to_whitened(init_params_phys)

        initial_state = blackjax.mcmc.mclmc.init(position=init_white, logdensity_fn=logdensity_fn, rng_key=init_key)

        def kernel_factory(inverse_mass_matrix):
            return blackjax.mcmc.mclmc.build_kernel(
                logdensity_fn=logdensity_fn,
                integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                inverse_mass_matrix=inverse_mass_matrix,
            )

        print(
            f"\nTuning MCLMC hyperparameters (single chain): num_steps={self.config.mclmc_num_warmup}, desired_energy_var={self.config.mclmc_desired_energy_var} ..."
        )
        t0 = time.time()
        tuned_state, tuned_params, _ = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel_factory,
            num_steps=self.config.mclmc_num_warmup,
            state=initial_state,
            rng_key=tune_key,
            diagonal_preconditioning=False,
            desired_energy_var=self.config.mclmc_desired_energy_var,
        )
        t_tune = time.time() - t0
        print(f"  Tuning done in {t_tune:0.2f}s")
        print(f"  Tuned L       : {float(tuned_params.L):.6g}")
        print(f"  Tuned step_size: {float(tuned_params.step_size):.6g}")

        sampling_alg = blackjax.mclmc(
            logdensity_fn,
            L=tuned_params.L,
            step_size=tuned_params.step_size,
        )

        def transform(state, info):
            return state.position

        print(f"\nRunning MCLMC chain for {self.config.mclmc_num_samples} steps ...")
        t0 = time.time()
        _, samples_white = blackjax.util.run_inference_algorithm(
            rng_key=run_key,
            initial_state=tuned_state,
            inference_algorithm=sampling_alg,
            num_steps=self.config.mclmc_num_samples,
            transform=transform,
            progress_bar=True,  # prints progress like the docs demo
        )
        t_run = time.time() - t0
        print(f"Sampling complete in {t_run:0.2f}s.")

        samples_white = np.asarray(samples_white)  # (T, D)
        samples_phys = jax.vmap(self.from_whitened)(samples_white)  # PyTree of arrays with leading dim T
        samples_phys = jax.device_get(samples_phys)

        return {
            "samples_white": samples_white,
            "samples_phys": samples_phys,  # may be None if mapping fails; keep it simple
            "tuned_params": {
                "L": float(tuned_params.L),
                "step_size": float(tuned_params.step_size),
            },
        }

    def calculate_individual_chi2s(self, params_phys: Dict) -> jnp.ndarray:
        """Calculate chi2 for each source individually."""
        obj_builder = ObjectiveFunctions(self.config, self.state, self.beam_models)

        if self.config.chi2_method == "fourier":

            def chi2_fn(y, x, f, d, n):
                return obj_builder._chi2_fourier_single(params_phys["beams"], y, x, f, d, n)

            chi2s = jax.vmap(chi2_fn, in_axes=(0, 0, 0, 0, 0))(
                params_phys["sources"]["yoff"],
                params_phys["sources"]["xoff"],
                params_phys["sources"]["flux"],
                self.state.maps_fft_jax,
                self.state.precision_jax,
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

        print(f"Starting NUTS sampling: {num_chains} chains, {self.config.nuts_num_warmup} warmup, {self.config.nuts_num_samples} samples")

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
            target_accept_prob=self.config.nuts_target_accept,
            max_tree_depth=self.config.nuts_max_tree_depth,
        )

        mcmc = MCMC(
            kernel,
            num_warmup=self.config.nuts_num_warmup,
            num_samples=self.config.nuts_num_samples,
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
