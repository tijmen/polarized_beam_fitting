"""
Polarized beam fitter implementation.

Contains the PolarizedBeamFitter class that handles both ML optimization
and NUTS sampling, supports single and multi-band configurations, and provides
efficient parallelization across devices.
"""

import copy
import time
import warnings
from typing import Dict, Optional, Tuple

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import optax

from .beam_model import create_beam_model
from .cache import CacheManager
from .data_loader import DataLoader
from .utils import (
    build_whitening_transform,
    calculate_tod_nyquist_radial_mask_smooth,
    check_convergence,
    compute_rectangular_ell_cut_indices,
    init_convergence_state,
    make_apodization_mask,
    params_from_logit,
    params_to_logit,
)


class ObjectiveFunctions:
    """Encapsulates objective function logic."""

    def __init__(self, config, fitter: "PolarizedBeamFitter", beam_models: Dict):
        self.config = config
        self.fitter = fitter
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
        T_only = getattr(self.config, "fit_T_only", False)
        vmap_chi2 = jax.vmap(
            lambda beam_params_list, yoff, xoff, flux, data_fft, precision: self._chi2_fourier_single(
                beam_params_list,
                yoff,
                xoff,
                flux,
                data_fft,
                precision,
                T_only=T_only,
            ),
            in_axes=(None, 0, 0, 0, 0, 0),
        )

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
        T_only = getattr(self.config, "fit_T_only", False)
        vmap_chi2 = jax.vmap(
            lambda beam_params_list, yoff, xoff, flux, data, weight: self._chi2_real_single(
                beam_params_list,
                yoff,
                xoff,
                flux,
                data,
                weight,
                T_only=T_only,
            ),
            in_axes=(None, 0, 0, 0, 0, 0),
        )

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

    def _chi2_real_single(self, beam_params_list, yoff, xoff, flux, data, weight, T_only=False):
        """Chi2 for single source in real space."""
        model = self._build_model(beam_params_list, yoff, xoff, flux)
        if T_only:
            data = data[..., :1]
            model = model[..., :1]
            weight = weight[..., :1, :1]
        residual = data - model
        chi2 = jnp.einsum("...i,...ij,...j->...", residual, weight, residual)
        return jnp.sum(chi2)

    def _chi2_fourier_single(self, beam_params_list, yoff, xoff, flux, data_fft, precision, T_only=False):
        """Chi2 for single source in Fourier space."""
        model = self._build_model(beam_params_list, yoff, xoff, flux)
        model_apod = model * self.fitter.apod_mask_broadcast
        model_fft = jnp.fft.fft2(model_apod, axes=(0, 1))
        if self.fitter.idx_y is not None and self.fitter.idx_x is not None:
            model_fft = jnp.take(model_fft, self.fitter.idx_y, axis=0)
            model_fft = jnp.take(model_fft, self.fitter.idx_x, axis=1)
        if T_only:
            data_fft = data_fft[..., :1]
            model_fft = model_fft[..., :1]
            precision = precision[..., :1, :, :1]
        residual_fft = data_fft - model_fft

        # Multi-band precision matrices arrive per source with axes (ny, nx, n_bands, n_stokes, n_bands, n_stokes)
        ny, nx = residual_fft.shape[:2]
        n_bands = residual_fft.shape[-2]
        n_stokes = residual_fft.shape[-1]

        residual_vec = residual_fft.reshape(ny, nx, n_bands * n_stokes)
        precision_matrix = precision.reshape(ny, nx, n_bands * n_stokes, n_bands * n_stokes)
        weighted_vec = jnp.einsum("yxvw,yxw->yxv", precision_matrix, residual_vec)
        chi2_per_k = jnp.einsum("yxv,yxv->yx", jnp.conj(residual_vec), weighted_vec)
        return jnp.sum(jnp.real(chi2_per_k))

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


def marginalized_linear_flux_chi2(data, template, weight):
    """Weighted chi2 with a single linear amplitude profiled out."""
    numerator = jnp.sum(weight * data * template)
    denominator = jnp.sum(weight * template * template)
    flux = numerator / denominator
    residual = data - flux * template
    chi2 = jnp.sum(weight * residual * residual)
    return chi2, flux


def peak_misnorm_fraction(band_flux, template, data_peak):
    """Fitted amplitude times template peak, divided by data peak."""
    return band_flux * jnp.max(template) / data_peak


class PolarizedBeamFitter:
    """
    Refactored polarized beam fitting class.

    Cleaner interface and more modular design while maintaining
    compatibility with existing beam_model and precision modules.
    """

    def __init__(self, config):
        """Initialize the fitter with configuration."""
        self.config = config
        self._setup_jax()
        self._print_welcome()

        # Initialize components
        self._initialize_state()
        self.beam_models = None
        self._cache_manager = None
        self._prepared_data_cache = None
        self._last_opt_state = None
        self._last_opt_variant = None

        # Load data
        self._load_data()
        self._configure_fit_context()

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

    def _initialize_state(self) -> None:
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

        # Coordinate grids and masks
        self.y_grid = y_grid
        self.x_grid = x_grid
        self.apod_mask = apod_mask_jax
        self.apod_mask_broadcast = apod_mask_jax[:, :, None, None]

        # Per-source data placeholders
        self.maps_jax: Optional[jnp.ndarray] = None
        self.weights_jax: Optional[jnp.ndarray] = None
        self.maps_fft_jax: Optional[jnp.ndarray] = None
        self.precision_jax: Optional[jnp.ndarray] = None
        self.idx_y, self.idx_x = compute_rectangular_ell_cut_indices((ny, nx), self.config.reso_arcmin, self.config.ellmax)
        self.raw_maps_numpy: Optional[np.ndarray] = None
        self.precision_numpy: Optional[np.ndarray] = None

        # Source metadata
        self.source_ids: Optional[np.ndarray] = None
        self.n_src: int = 0
        self.fields: Optional[np.ndarray] = None
        self.n_bands: int = len(self.config.bands)
        self.init_yoff: Optional[np.ndarray] = None
        self.init_xoff: Optional[np.ndarray] = None
        self.init_flux: Optional[np.ndarray] = None

    def _create_beam_models(self) -> Dict:
        """Create beam models for each band."""
        print("Creating beam models...")
        models = {}
        for band in self.config.bands:
            models[band] = create_beam_model(self.config, self.y_grid, self.x_grid, band)
        return models

    def _configure_fit_context(self):
        """Build beam models, objective, compiled loss, and initial parameters."""
        self.beam_models = self._create_beam_models()
        self.objective_function = ObjectiveFunctions(self.config, self, self.beam_models).build_objective()
        self._loss_and_grad = jax.jit(jax.value_and_grad(lambda params_logit, data: self.objective_function(params_logit, data, None)))
        self.params_physical = self._initialize_parameters()
        self.params_logit = params_to_logit(self.params_physical, self.config)

    def _load_data(self):
        """Load and prepare data using cache if available."""
        print("Loading data...")

        cache = CacheManager(self.config)
        loader_class = self.config.data_loader_class or DataLoader
        loader = loader_class(self.config)

        data = cache.load()
        cache_hit = data is not None
        if data is None:
            data = loader.load_and_prepare()
        data_list = list(data)

        self._cache_manager = cache
        self._assign_prepared_data(data_list)

        if not cache_hit:
            self._add_gaussian_source_prefit_to_cache(data_list)
            self._assign_prepared_data(data_list)
            cache.save(tuple(data_list))

    def _assign_prepared_data(self, data_list):
        """Install a prepared-data tuple/list into this fitter's data state."""
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
            precision,
            debug_precision,
        ) = data_list

        self._prepared_data_cache = data_list

        # Store in state
        self.source_ids = source_ids
        self.n_src = n_src
        self.init_yoff = gaussfit_yoff
        self.init_xoff = gaussfit_xoff
        self.init_flux = gaussfit_amp
        self.fields = source_fields
        self.raw_maps_numpy = raw_maps
        self.precision_numpy = precision
        self.debug_precision = debug_precision

        # Convert to JAX arrays
        self.maps_jax = jnp.asarray(maps, dtype=self.config.dtype_jax_real)

        # Setup for specific chi2 method
        if self.config.chi2_method == "fourier":
            self._setup_fourier_data(maps_fft, precision)
        else:
            self._setup_real_space_data(weights)

    def _add_gaussian_source_prefit_to_cache(self, data_list):
        """Run a Gaussian, T-only source prefit and store y/x/T initial values."""
        print("Preparing Gaussian T-only source-parameter cache...")
        original_state = (
            self.config,
            self.beam_models,
            getattr(self, "objective_function", None),
            getattr(self, "_loss_and_grad", None),
            getattr(self, "params_physical", None),
            getattr(self, "params_logit", None),
            self._last_opt_state,
            self._last_opt_variant,
        )

        try:
            prefit_config = copy.copy(self.config)
            prefit_config.beam_model_type = "gaussian"
            prefit_config.fit_T_only = True
            prefit_config.refit_position = True
            prefit_config.refit_Tflux = True
            prefit_config.solver = "tuned"

            self.config = prefit_config
            self._last_opt_state = None
            self._last_opt_variant = None
            self._configure_fit_context()

            best_fit = self.run_fit()
        finally:
            (
                self.config,
                self.beam_models,
                self.objective_function,
                self._loss_and_grad,
                self.params_physical,
                self.params_logit,
                self._last_opt_state,
                self._last_opt_variant,
            ) = original_state

        data_list[0] = np.asarray(best_fit["sources"]["yoff"], dtype=self.config.dtype_np_real)
        data_list[1] = np.asarray(best_fit["sources"]["xoff"], dtype=self.config.dtype_np_real)
        data_list[2] = np.asarray(data_list[2], dtype=self.config.dtype_np_real).copy()
        data_list[2][..., 0] = np.asarray(best_fit["sources"]["flux"][..., 0], dtype=self.config.dtype_np_real)
        print("Gaussian T-only source parameters added to prepared-data cache.")

    def _setup_fourier_data(self, maps_fft, precision: np.ndarray):
        """Setup cached Fourier-space arrays for optimization."""
        if maps_fft is None:
            raise ValueError("Fourier-space chi2 requested but cached FFT maps are missing.")

        self.maps_fft_jax = jnp.asarray(maps_fft, dtype=self.config.dtype_jax_complex)
        self.precision_jax = jnp.asarray(precision, dtype=self.config.dtype_jax_real)
        self.objective_data = (self.maps_fft_jax, self.precision_jax)

    def _setup_real_space_data(self, weights):
        """Setup data for real-space analysis."""
        self.weights_jax = jnp.asarray(weights, dtype=self.config.dtype_jax_real)
        self.objective_data = (self.maps_jax, self.weights_jax)

    def _calculate_nyquist_mask(self, source_ids):
        """Return radial Fourier mask per source (values in [0, 1])."""
        map_shape = tuple(int(dim) for dim in self.apod_mask.shape)
        masks = [calculate_tod_nyquist_radial_mask_smooth(sid, map_shape, self.config) for sid in source_ids]
        return np.stack(masks)  # (n_src, ny, nx)

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
                )  # yes, jax.tree.map really is correct for our version of python. NEVER CHANGE THIS TO tree_util, which doesn't exist
            )

        # Initialize source parameters
        if self.init_yoff is None or self.init_xoff is None or self.init_flux is None:
            raise ValueError("Initial source parameters are missing; ensure data loading completed successfully.")

        params["sources"] = {
            "yoff": jnp.asarray(self.init_yoff, dtype=self.config.dtype_jax_real),
            "xoff": jnp.asarray(self.init_xoff, dtype=self.config.dtype_jax_real),
            "flux": jnp.asarray(self.init_flux, dtype=self.config.dtype_jax_real),
        }

        return params

    def _uses_reduced_adam_parameter_space(self):
        return not (getattr(self.config, "refit_position", True) and getattr(self.config, "refit_Tflux", True))

    def _adam_fit_params_from_full(self, params_logit):
        if not self._uses_reduced_adam_parameter_space():
            return params_logit

        fit_params = {"beams": params_logit["beams"], "sources": {}}
        if getattr(self.config, "refit_position", True):
            fit_params["sources"]["yoff"] = params_logit["sources"]["yoff"]
            fit_params["sources"]["xoff"] = params_logit["sources"]["xoff"]
        if getattr(self.config, "refit_Tflux", True):
            fit_params["sources"]["flux"] = params_logit["sources"]["flux"]
        else:
            fit_params["sources"]["flux_QU"] = params_logit["sources"]["flux"][..., 1:]
        return fit_params

    def _full_params_from_adam_fit_params(self, fit_params_logit, fixed_params_logit):
        if not self._uses_reduced_adam_parameter_space():
            return fit_params_logit

        full_params = {"beams": fit_params_logit["beams"], "sources": {}}
        if getattr(self.config, "refit_position", True):
            full_params["sources"]["yoff"] = fit_params_logit["sources"]["yoff"]
            full_params["sources"]["xoff"] = fit_params_logit["sources"]["xoff"]
        else:
            full_params["sources"]["yoff"] = fixed_params_logit["sources"]["yoff"]
            full_params["sources"]["xoff"] = fixed_params_logit["sources"]["xoff"]

        if getattr(self.config, "refit_Tflux", True):
            full_params["sources"]["flux"] = fit_params_logit["sources"]["flux"]
        else:
            full_params["sources"]["flux"] = jnp.concatenate(
                [fixed_params_logit["sources"]["flux"][..., :1], fit_params_logit["sources"]["flux_QU"]],
                axis=-1,
            )
        return full_params

    def _iter_final_logit_physical_bounds(self):
        active_model_bounds = self.config.active_beam_model_bounds
        for band_idx, (beam_logit, beam_physical) in enumerate(zip(self.params_logit["beams"], self.params_physical["beams"])):
            for param_name, logit_value in beam_logit.items():
                yield (
                    f"params_logit.beams[{band_idx}].{param_name}",
                    logit_value,
                    beam_physical[param_name],
                    active_model_bounds[param_name],
                )

        yoff_bounds, xoff_bounds = self.config.source_position_bounds
        source_bounds = {
            "yoff": yoff_bounds,
            "xoff": xoff_bounds,
            "flux": self.config.source_flux_bounds,
        }
        for param_name, bounds in source_bounds.items():
            yield (
                f"params_logit.sources.{param_name}",
                self.params_logit["sources"][param_name],
                self.params_physical["sources"][param_name],
                bounds,
            )

    def _warn_if_final_logit_out_of_range(self):
        offending_params = []
        total_offending = 0
        for name, logit_value, physical_value, bounds in self._iter_final_logit_physical_bounds():
            logit_values = np.asarray(jax.device_get(logit_value))
            if logit_values.size == 0:
                continue
            mask = (logit_values < -50.0) | (logit_values > 50.0)
            if not np.any(mask):
                continue

            physical_values = np.asarray(jax.device_get(physical_value))
            lower, upper = bounds
            lower = np.broadcast_to(np.asarray(jax.device_get(lower)), physical_values.shape)
            upper = np.broadcast_to(np.asarray(jax.device_get(upper)), physical_values.shape)

            offending_logit_values = logit_values[mask]
            offending_physical_values = physical_values[mask]
            offending_lower = lower[mask]
            offending_upper = upper[mask]
            total_offending += int(offending_logit_values.size)
            offending_params.append(
                f"{name}: count={offending_logit_values.size}, "
                f"logit=[{np.min(offending_logit_values):.3g}, {np.max(offending_logit_values):.3g}], "
                f"physical=[{np.min(offending_physical_values):.6g}, {np.max(offending_physical_values):.6g}], "
                f"bounds=[{np.min(offending_lower):.6g}, {np.max(offending_upper):.6g}]"
            )

        if offending_params:
            warnings.warn(
                "Final best-fit logit parameters outside [-50, 50]: "
                + "; ".join(offending_params)
                + f". Total offending entries: {total_offending}. "
                "These parameters are extremely close to their configured physical bounds.",
                RuntimeWarning,
                stacklevel=2,
            )

    def run_fit(self):
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
        elif self.config.solver == "tuned":
            self._run_tuned_optimization()
        else:
            raise ValueError(f"Unknown solver: {self.config.solver}")

        self._warn_if_final_logit_out_of_range()
        return self.params_physical

    def _run_bfgs(self):
        """Run BFGS optimization."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"jax\.interpreters\.batching\.NotMapped is deprecated\.", category=DeprecationWarning
            )
            import optimistix as optx

        solver = optx.BFGS(**self.config.bfgs_kwargs)
        use_reduced_space = self._uses_reduced_adam_parameter_space()
        if use_reduced_space:
            fixed_params_logit = self.params_logit
            fit_params_logit = self._adam_fit_params_from_full(self.params_logit)

            def objective_fit_params(params_logit_fit, args, extra_args=None):
                fixed_params_logit, data = args
                params_logit_full = self._full_params_from_adam_fit_params(params_logit_fit, fixed_params_logit)
                return self.objective_function(params_logit_full, data, extra_args)

            objective_args = (fixed_params_logit, self.objective_data)
        else:
            fit_params_logit = self.params_logit
            fixed_params_logit = None
            objective_fit_params = self.objective_function
            objective_args = self.objective_data

        sol = optx.minimise(
            objective_fit_params,
            solver,
            fit_params_logit,
            args=objective_args,
            max_steps=self.config.n_steps,
            throw=False,
        )

        if sol.result != optx.RESULTS.successful:
            raise RuntimeError(f"BFGS failed: {optx.RESULTS[sol.result]}")

        self.params_logit = self._full_params_from_adam_fit_params(sol.value, fixed_params_logit) if use_reduced_space else sol.value
        self.params_physical = params_from_logit(self.params_logit, self.config)

        print(f"Optimization finished after {sol.stats['num_steps']} steps.")

    def _run_adam(self):
        """
        Run Adam (or compatible Optax) optimization.

        My recommendation is to run Adam first with a more aggressive learning rate, then
        switch to AMSGrad for fine-tuning, as Adam converges faster, but can bounce back out
        of local minima after some time.
        """
        variant = self.config.adam_variant
        print(f"Starting optimization with optax variant {variant}.")
        use_reduced_space = self._uses_reduced_adam_parameter_space()
        if use_reduced_space:
            fixed_params_logit = self.params_logit
            fit_params_logit = self._adam_fit_params_from_full(self.params_logit)

            def objective_fit_params(params_logit_fit, fixed_params_logit, data):
                params_logit_full = self._full_params_from_adam_fit_params(params_logit_fit, fixed_params_logit)
                return self.objective_function(params_logit_full, data, None)

            loss_and_grad = jax.jit(jax.value_and_grad(objective_fit_params))

            def evaluate_loss_and_grad(params_logit_fit):
                return loss_and_grad(params_logit_fit, fixed_params_logit, self.objective_data)

        else:
            fit_params_logit = self.params_logit
            fixed_params_logit = None
            loss_and_grad = self._loss_and_grad

            def evaluate_loss_and_grad(params_logit_fit):
                return loss_and_grad(params_logit_fit, self.objective_data)

        optimizer = getattr(optax, variant)(**self.config.adam_kwargs)
        opt_state = optimizer.init(fit_params_logit)

        if self._last_opt_state is not None:
            print("Continuing from previous optimizer state.")
            s_prev = self._last_opt_state[0]  # has .count, .mu, .nu
            s_cur = opt_state[0]
            kwargs = dict(count=s_prev.count, mu=s_prev.mu, nu=s_prev.nu)
            s_new0 = s_cur._replace(**kwargs)
            opt_state = (s_new0,) + opt_state[1:]

        # Initialize convergence tracking
        convergence_state = init_convergence_state()
        initial_loss, initial_grads = evaluate_loss_and_grad(fit_params_logit)
        initial_grad_norm = optax.global_norm(initial_grads)
        print(f"Initial loss: {initial_loss:.2f}, Initial gradient norm: {initial_grad_norm:.2f}")

        # Initialize optimization history if debug is enabled
        if self.config.debug:
            self.opt_history = []
            print("Debug mode: storing optimization history in fitter.opt_history")

        for i in range(self.config.n_steps):
            loss, grads = evaluate_loss_and_grad(fit_params_logit)
            grad_norm = optax.global_norm(grads)

            # Store history if debug is enabled
            if self.config.debug:
                params_logit_full = self._full_params_from_adam_fit_params(fit_params_logit, fixed_params_logit)
                params_phys = params_from_logit(params_logit_full, self.config)
                history_entry = {
                    "step": i,
                    "loss": float(loss),
                    "grad_norm": float(grad_norm),
                    "params_physical": jax.device_get(params_phys),
                    "params_logit": jax.device_get(params_logit_full),
                    "gradients": jax.device_get(grads),
                    "mu": jax.device_get(opt_state[0].mu) if hasattr(opt_state[0], "mu") else None,
                    "nu": jax.device_get(opt_state[0].nu) if hasattr(opt_state[0], "nu") else None,
                    "count": int(opt_state[0].count) if hasattr(opt_state[0], "count") else None,
                }
                self.opt_history.append(history_entry)

            # Check convergence and track best parameters
            converged, message, convergence_state = check_convergence(
                loss,
                grad_norm,
                i,
                self.config,
                convergence_state,
                initial_grad_norm,
                fit_params_logit,
            )
            if converged:
                print(f"Converged at step {i}: {message}")
                if self.config.convergence_criterion == "loss_history":
                    print(f"Returning best loss found: {convergence_state['best_loss']:.2f}")
                break

            updates, opt_state = optimizer.update(grads, opt_state)
            fit_params_logit = optax.apply_updates(fit_params_logit, updates)
            if i % 10 == 0:  # Print every 10 steps
                print(f"Step {i}/{self.config.n_steps}: loss={loss:.2f}, |grad|={grad_norm:.2f}")

        # Use best parameters if available, otherwise use current parameters
        if convergence_state["best_params"] is not None:
            fit_params_logit = convergence_state["best_params"]
        self.params_logit = (
            self._full_params_from_adam_fit_params(fit_params_logit, fixed_params_logit) if use_reduced_space else fit_params_logit
        )
        self.params_physical = params_from_logit(self.params_logit, self.config)

        # Remember optimizer state/variant for possible warm-start next time
        self._last_opt_state = opt_state
        self._last_opt_variant = variant

    def _run_tuned_optimization(self):
        """
        Wrapper for _run_adam overwriting configuration with pre-determined
        hyperparameters.

        I tuned the optimization procedure. Most critical is that Adam over the logit
        parameters has a major problem. By the time some of the low-curvature parameters
        (such as polarization beam parameters) are optimized, some of the hyper-sensitive
        parameters such as source positions or temperature flux parameters have been sitting
        in their minimum for so long that the optimizer state has accumulated to a point where
        it exceeds alpha k > 2 and the optimizer explodes, followed by rapid recovery.

        This tuned procedure first aggressively allows this phenomenon to occur, then
        detects it and finishes the optimization with a gentle and robust completion using amsgrad.
        """
        self.config.adam_variant = "adam"  # start with aggressive Adam
        self.config.adam_kwargs = {"learning_rate": 5e-3}  # aggressive learning rate
        self.config.loss_history_length = 100  # go until Adam isn't improving anymore
        self.config.n_steps = 8000  # the longest we're willing to wait
        self._run_adam()
        self.config.adam_variant = "amsgrad"  # switch to gentle AMSGrad
        self.config.n_steps = 1000  # the longest we're willing to wait for the fine-tuning
        self.config.adam_kwargs = {"learning_rate": 1e-3}  # gentle learning rate
        self.config.loss_history_length = 10  # go until AMSGrad isn't improving anymore
        self._run_adam()

    def sample_with_mclmc(self):
        """
        Single-chain MCLMC
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="JAXopt is no longer maintained.*", category=DeprecationWarning)
            import blackjax

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
        obj_builder = ObjectiveFunctions(self.config, self, self.beam_models)

        if self.config.chi2_method == "fourier":

            def chi2_fn(y, x, f, d, n):
                return obj_builder._chi2_fourier_single(params_phys["beams"], y, x, f, d, n)

            chi2s = jax.vmap(chi2_fn, in_axes=(0, 0, 0, 0, 0))(
                params_phys["sources"]["yoff"],
                params_phys["sources"]["xoff"],
                params_phys["sources"]["flux"],
                self.maps_fft_jax,
                self.precision_jax,
            )
        else:

            def chi2_fn(y, x, f, d, w):
                return obj_builder._chi2_real_single(params_phys["beams"], y, x, f, d, w)

            chi2s = jax.vmap(chi2_fn, in_axes=(0, 0, 0, 0, 0))(
                params_phys["sources"]["yoff"],
                params_phys["sources"]["xoff"],
                params_phys["sources"]["flux"],
                self.maps_jax,
                self.weights_jax,
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
        diag_hessian_flat_np = np.asarray(jax.device_get(diag_hessian_flat))
        nonfinite_mask = ~np.isfinite(diag_hessian_flat_np)
        if np.any(nonfinite_mask):
            bad_indices = np.flatnonzero(nonfinite_mask)
            bad_preview = ", ".join(str(int(idx)) for idx in bad_indices[:10])
            raise FloatingPointError(
                "Non-finite diagonal Hessian entries encountered while preparing "
                f"the whitening transform: {bad_indices.size}/{diag_hessian_flat_np.size} "
                f"entries are non-finite. First bad flattened indices: {bad_preview}."
            )

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
        from numpyro.infer import MCMC, NUTS

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
        obj_builder = ObjectiveFunctions(self.config, self, self.beam_models)

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
