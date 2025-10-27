"""
Bootstrap resampling for polarized beam fitting.
Uses a warm-start procedure where we first fit all sources normally, then
resample with replacement, starting the optimizer at the all-sources-once maximum likelihood solution.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .fitter import PolarizedBeamFitter
from .utils import params_to_logit


class BootstrapBeamFitter:
    """Bootstrap wrapper for PolarizedBeamFitter"""

    def __init__(self, config):
        self.config = config
        self.base_fitter = PolarizedBeamFitter(config=self.config)
        self.ml_params_physical = None

    def run_fit(self):
        """Run fitting with bootstrap uncertainty estimation."""
        print("=== Starting Polarized Beam Fitting ===")

        print("\n1. Running initial maximum likelihood fit...")
        ml_fit_results = self.base_fitter.run_fit()
        self.ml_params_physical = self.base_fitter.params_physical

        ml_fit_results["total_chi2"] = np.sum(self.base_fitter.latest_chi2s)

        print(f"\n2. Running {self.config.n_bootstrap_samples} bootstrap iterations...")
        bootstrap_params = self._run_bootstrap_fits()

        print(f"\nCompleted {len(bootstrap_params)} bootstrap samples")
        return {
            "ml_fit": ml_fit_results,
            "bootstrap_fits": bootstrap_params,
        }

    def _run_bootstrap_fits(self):
        """Run bootstrap fits by resampling sources and reusing the base fitter."""
        n_sources = len(self.base_fitter.source_ids)
        rng_key = jax.random.PRNGKey(self.config.bootstrap_seed)

        # Generate bootstrap sample indices
        rng_key, subkey = jax.random.split(rng_key)
        bootstrap_indices = jax.random.choice(
            subkey,
            n_sources,
            shape=(self.config.n_bootstrap_samples, n_sources),
            replace=True,
        )

        # Get original data arrays from cache
        data_cache = self.base_fitter._prepared_data_cache
        (
            gaussfit_yoff,
            gaussfit_xoff,
            gaussfit_amp,
            raw_maps,
            _,  # qu_templates - not needed for bootstrap
            maps,
            weights,
            maps_fft,
            source_ids,
            source_fields,
            _,  # n_src - will compute from indices
            precision,
            _,  # debug_precision - not needed
        ) = data_cache

        bootstrap_params = []

        for i, indices in enumerate(bootstrap_indices):
            print(f"Bootstrap iteration {i + 1}/{self.config.n_bootstrap_samples}")

            # Resample all source-specific arrays
            indices_np = np.array(indices, dtype=int)

            # Update fitter with resampled data
            self._update_fitter_data(indices_np, maps, weights, maps_fft, precision)

            # Initialize parameters: beam from ML, sources from resampled ML values
            self.base_fitter.params_physical = self._create_bootstrap_initial_params(indices_np)
            self.base_fitter.params_logit = params_to_logit(self.base_fitter.params_physical, self.config)

            # Note: we leave the optimizer state. This is a weird thing to do, so might consider resetting it...
            # self.base_fitter._last_opt_state = None

            self.base_fitter.run_fit()

            bootstrap_params.append(self.base_fitter.params_physical)

        return bootstrap_params

    def _update_fitter_data(self, indices_np, maps, weights, maps_fft, precision):
        """Update the base fitter with resampled data."""
        # Resample the arrays
        maps_resampled = maps[indices_np]

        # Update JAX arrays in fitter
        self.base_fitter.maps_jax = jnp.asarray(maps_resampled, dtype=self.config.dtype_jax_real)

        if self.config.chi2_method == "fourier":
            maps_fft_resampled = maps_fft[indices_np]
            precision_resampled = precision[indices_np]
            self.base_fitter.maps_fft_jax = jnp.asarray(maps_fft_resampled, dtype=self.config.dtype_jax_complex)
            self.base_fitter.precision_jax = jnp.asarray(precision_resampled, dtype=self.config.dtype_jax_real)
            self.base_fitter.objective_data = (self.base_fitter.maps_fft_jax, self.base_fitter.precision_jax)
        else:
            weights_resampled = weights[indices_np]
            self.base_fitter.weights_jax = jnp.asarray(weights_resampled, dtype=self.config.dtype_jax_real)
            self.base_fitter.objective_data = (self.base_fitter.maps_jax, self.base_fitter.weights_jax)

    def _create_bootstrap_initial_params(self, indices):
        """
        Create initial parameters for bootstrap sample.
        Beam params from ML, source params duplicated according to resampling.
        """
        ml_params = self.ml_params_physical

        # Beam parameters: use ML solution
        bootstrap_params = {"beams": ml_params["beams"], "sources": {}}

        # Source parameters: duplicate according to bootstrap indices
        for key in ["yoff", "xoff", "flux"]:
            bootstrap_params["sources"][key] = ml_params["sources"][key][indices]

        return bootstrap_params
