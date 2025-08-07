"""
Bootstrap uncertainty estimation for polarized beam fitting.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from .fitter import PolarizedBeamFitter
from .utils import params_from_logit


def make_bootstrap_objective(get_individual_chi2s_func, weight_array):
    """
    Creates and JIT-compiles a bootstrap objective function with a specific
    bootstrap weight array baked into its closure.
    """

    @jax.jit
    def bootstrap_objective(params, extra_args=None):
        individual_chi2s = get_individual_chi2s_func(params)
        return jnp.sum(individual_chi2s * weight_array)

    return bootstrap_objective


class BootstrapBeamFitter:
    """
    Bootstrap wrapper for PolarizedBeamFitter. The plan is:

    1. Initialize the fitter as usual, including data loading and leakage template construction and subtraction.

    2. Run the fitter to get the maximum likelihood results.

    3. Prepare `bootstrap_weight` arrays. There are `num_bootstrap` of these, each with `num_sources` elements.
    They contain numbers like [2,0,0,0,1,1,0,1,0,1,3,1,0,...] which indicate how many copies of
    that source were selected during sampling with replacement.

    4. For each bootstrap weight array, we run the fitter again, modifying
    the objective function with to be the dot product of the individual per-source chi2 values with the bootstrap weight array.

    5. We collect the best-fit parameters from each bootstrap sample, save it, and calculate statistics over it.
    """

    def __init__(self, config):
        self.config = config
        self.base_fitter = PolarizedBeamFitter(config=self.config)
        self.original_fit_results = None

    def run_fit(self):
        """Run fitting with optional bootstrap uncertainty estimation."""
        print("=== Starting Polarized Beam Fitting ===")

        # Step 1 & 2: Original fit
        print("\n1. Running initial maximum likelihood fit...")
        self.original_fit_results = self.base_fitter.run_fit()

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

        # Step 5: Analyze results
        print("\n3. Analyzing bootstrap results...")
        return self._analyze_results(bootstrap_params)

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

    def _run_bootstrap_fits(self):
        """Run all bootstrap fits. Each fit re-optimises the parameters using
        a bootstrap-specific objective that is the dot product of per-source
        chi2 values with the corresponding bootstrap weight array. We warm-start
        every iteration from the maximum-likelihood parameters obtained in the
        initial fit."""

        bootstrap_weights = self._prepare_bootstrap_weights()

        # Choose initialization strategy based on config
        if self.config.bootstrap_warm_start:
            # Warm-start: use the logit-space parameters from the original ML solution as the
            # starting point for every bootstrap iteration, then update after each iteration
            initial_params_logit = jax.tree_util.tree_map(lambda x: x, self.base_fitter.params_logit)
            print("Using warm-start strategy: each iteration starts from previous optimum")
        else:
            # Cold-start: use the same initial parameters that the global fit started from
            print("Warning: cold-start is not implemented. Switching to warm-start approach...")
            initial_params_logit = jax.tree_util.tree_map(lambda x: x, self.base_fitter.params_logit)
            print("Using cold-start strategy: each iteration starts from same initial params as global fit")

        bootstrap_params = []

        for i, weight_array in enumerate(bootstrap_weights):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Bootstrap iteration {i + 1}/{self.config.n_bootstrap_samples}")

            # Build a dedicated objective function for this bootstrap sample.
            objective_func = make_bootstrap_objective(self.base_fitter._get_individual_chi2s, weight_array)

            solver = optx.BFGS(rtol=1e-24, atol=1e-24)

            sol = optx.minimise(
                objective_func,
                solver,
                initial_params_logit,
                max_steps=self.config.n_steps,
                throw=False,
            )

            if sol.result != optx.RESULTS.successful:
                raise RuntimeError(f"BFGS failed: {optx.RESULTS[sol.result]} after {sol.stats['num_steps']} steps")

            # Convert logit parameters back to physical space for storage
            physical_params = params_from_logit(sol.value, self.config)

            # Compute the final weighted chi2 for this bootstrap sample
            bootstrap_chi2 = float(jnp.sum(self.base_fitter._get_individual_chi2s(physical_params) * weight_array))
            physical_params["bootstrap_chi2"] = bootstrap_chi2

            bootstrap_params.append(physical_params)

            # Update starting point for next iteration only if using warm-start strategy
            if self.config.bootstrap_warm_start:
                initial_params_logit = sol.value

        print(f"Completed {len(bootstrap_params)}/{self.config.n_bootstrap_samples} successful bootstrap iterations.")

        return bootstrap_params

    # Note: _convert_physical_to_logit is no longer needed with the new parameter structure

    def _analyze_results(self, bootstrap_params):
        """Analyze bootstrap results."""
        if not bootstrap_params:
            raise ValueError("No successful bootstrap iterations")

        # Organize parameters
        organized = self._organize_params(bootstrap_params)

        # Compute statistics
        results = {
            "original_fit": self.original_fit_results,
            "bootstrap_mean": {k: np.mean(v, axis=0) for k, v in organized.items()},
            "bootstrap_std": {k: np.std(v, axis=0, ddof=1) for k, v in organized.items()},
            "confidence_intervals": self._compute_confidence_intervals(organized),
            "n_successful_iterations": len(bootstrap_params),
            "n_requested_iterations": self.config.n_bootstrap_samples,
            "individual_fits": bootstrap_params,
        }

        return results

    def _organize_params(self, bootstrap_params):
        """Organize bootstrap parameters into arrays."""
        organized = {}

        # Handle beam parameters - extract from each band
        n_bands = len(self.config.bands)
        for band_idx in range(n_bands):
            band_key = f"band_{band_idx}"
            if self.config.beam_model_type == "beta_pol":
                organized[f"{band_key}_beta_pol"] = np.array([p["beams"][band_idx]["beta_pol"] for p in bootstrap_params])
            elif self.config.beam_model_type == "gaussian":
                organized[f"{band_key}_T_width_arcmin"] = np.array([p["beams"][band_idx]["T_width_arcmin"] for p in bootstrap_params])
                organized[f"{band_key}_P_width_arcmin"] = np.array([p["beams"][band_idx]["P_width_arcmin"] for p in bootstrap_params])
            elif self.config.beam_model_type == "bsplines":
                organized[f"{band_key}_T_coeffs"] = np.array([p["beams"][band_idx]["T_coeffs"] for p in bootstrap_params])
                organized[f"{band_key}_P_coeffs"] = np.array([p["beams"][band_idx]["P_coeffs"] for p in bootstrap_params])
            elif self.config.beam_model_type == "beta_T":
                organized[f"{band_key}_beta_T"] = np.array([p["beams"][band_idx]["beta_T"] for p in bootstrap_params])
            elif self.config.beam_model_type == "bsplines_plus_gaussian":
                organized[f"{band_key}_gaussian_sigma_arcmin"] = np.array([p["beams"][band_idx]["gaussian_sigma_arcmin"] for p in bootstrap_params])
                organized[f"{band_key}_bspline_coeffs_T"] = np.array([p["beams"][band_idx]["bspline_coeffs_T"] for p in bootstrap_params])
                organized[f"{band_key}_bspline_coeffs_P"] = np.array([p["beams"][band_idx]["bspline_coeffs_P"] for p in bootstrap_params])

        # Handle source parameters
        organized["sources_yoff"] = np.array([p["sources"]["yoff"] for p in bootstrap_params])
        organized["sources_xoff"] = np.array([p["sources"]["xoff"] for p in bootstrap_params])
        organized["sources_flux"] = np.array([p["sources"]["flux"] for p in bootstrap_params])

        return organized

    def _compute_confidence_intervals(self, organized):
        """Compute confidence intervals."""
        confidence_intervals = {}

        for confidence_level in self.config.bootstrap_confidence_levels:
            lower_percentile = (100 - confidence_level) / 2
            upper_percentile = 100 - lower_percentile

            ci_dict = {}
            for param_name, param_array in organized.items():
                ci_dict[param_name] = {
                    "lower": np.percentile(param_array, lower_percentile, axis=0),
                    "upper": np.percentile(param_array, upper_percentile, axis=0),
                }
            confidence_intervals[f"{confidence_level}%"] = ci_dict

        return confidence_intervals

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
    def noise_psd_calculator(self):
        return self.base_fitter.noise_psd_calculator

    @property
    def noise_psd_numpy(self):
        return np.array(self.base_fitter.state.noise_psd_jax) if self.base_fitter.state.noise_psd_jax is not None else None

    @property
    def maps_numpy(self):
        return np.array(self.base_fitter.state.maps_jax)
