"""
Configuration module for polarized beam fitting.

Contains all the constants and parameters used in the analysis.
"""

import os
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from spt3g import core


@dataclass
class BeamFittingConfig:
    """Configuration class for polarized beam fitting analysis."""

    # File paths
    coadd_filenames = [
        "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_tau_decon_winter.g3",
        "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_tau_decon_summera.g3",
        "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_tau_decon_summerb.g3",
        "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_tau_decon_summerc.g3",
        "/home/tijmen/cmb_analysis/beam_analysis/data/J1924-2914.g3",
        "/home/tijmen/cmb_analysis/beam_analysis/data/J2258-2758.g3",
    ]
    noise_psd_path = "/home/tijmen/cmb_analysis/beam_analysis/data/subfield_noise_PSD_{band}GHz_mean_sub2.fits"
    output_dir = "/home/tijmen/cmb_analysis/beam_analysis/output"
    cache_dir = "/home/tijmen/cmb_analysis/beam_analysis/cache"
    betapol_data_path = "/home/tijmen/cmb_analysis/beam_analysis/polarized_beam_fitting/data/betapol_TdH.npz"

    # Analysis parameters
    bands = ["90GHz"]  # ["90GHz", "150GHz", "220GHz"]  # Frequency bands for analysis
    double_precision = True  # Use 64-bit precision for all calculations
    solver = "optax_adam"  # "optimistix_bfgs", "optax_adam"
    bfgs_kwargs = {"atol": 1e-24, "rtol": 1e-24, "verbose": frozenset({"step_size", "loss"})}
    adam_kwargs = {"learning_rate": 0.001}
    # Convergence criteria options
    convergence_criterion = "loss_history"  # "absolute_gtol", "relative_gtol", or "loss_history"

    # Absolute gradient tolerance (used when convergence_criterion = "absolute_gtol")
    absolute_gtol = 100.0

    # Relative gradient tolerance (used when convergence_criterion = "relative_gtol")
    relative_gtol = 1.0 / (3e5)  # Default: 1 part in 300,000

    # Loss history parameters (used when convergence_criterion = "loss_history")
    loss_history_length = 10  # Number of steps without improvement before convergence

    debug = False

    # Map parameters
    map_size_pix = 300
    reso_arcmin = 0.1
    apodization_width_pix = 10

    # B-spline parameters
    knot_spacing_arcmin = 0.25
    spline_k = 4  # Cubic B-spline
    spline_rmax_arcmin = 10.0

    # Source selection criteria
    min_t_amplitude = 500 * core.G3Units.uK
    max_zero_fraction = 0.05
    exclude_from_leakage_template = None

    # Processing options
    leakage_weighting = "linear"  # "flat", "linear", "squared", or "median"

    # Chi-squared calculation method
    chi2_method = "fourier"  # "fourier" or "real_space"

    # Noise PSD options
    # Available options:
    #  'clusterfinder_psd': Load pre-computed clusterfinder instrument noise PSD from FITS file
    #  'kx_averaged': Calculate individual noise PSDs using k_x averaging with max heuristic
    #  'white_noise': Simple white noise assumption with constant PSD values
    #  'ensemble_asd_mean': Average amplitude spectral densities across sources then convert to PSD
    # This is ignored if chi2_method = "real_space"
    noise_psd_method = "white_noise"

    # Data-driven Noise PSD parameters (used when noise_psd_method = 'data_driven')
    # Calculates PSD directly from regions of the map away from the source
    # We start by creating a modified apodization mask with a configurable
    # hole in the center, then calculate the noise PSD from the empty regions.
    noise_hole_radius_arcmin = 4.0  # Radius of central hole for noise calculation (arcmin)

    # Optimization parameters
    n_steps = 100000

    # Bootstrap resampling parameters
    enable_bootstrap = False  # Enable/disable bootstrap uncertainty estimation
    n_bootstrap_samples = 100  # Number of bootstrap iterations
    bootstrap_seed = 42  # Random seed for reproducibility (None for random)
    bootstrap_confidence_levels = [68, 95]  # Confidence levels to report (percentiles)

    # Plotting options
    n_diagnostic_plots = 3  # Number of highest chi2 sources to plot diagnostics for.
    # Can be an integer (default 3), "all" to plot all sources,
    # or 0 to disable diagnostic plots entirely

    source_position_bounds = (
        (-5.0, 5.0),  # yoff (source center y offset in pixels)
        (-5.0, 5.0),  # xoff (source center x offset in pixels)
    )

    source_flux_bounds = (-5.0, 100.0)  # T, Q, U flux in uK

    # Select which can of parameterized model you want to fit.
    # The beta_pol model just has one free parameter
    # The Gaussian model frees the width of the T and P beams separately
    # The B-spline model is very flexible, but suffers from the P beam spike problem described at
    #   https://pole.uchicago.edu/spt3g/index.php/Polarized_Point_Source_Stack#An_Explanation_for_the_Peaky_P_Beam
    # The bsplines_plus_gaussian model combines a central Gaussian with area-normalized B-splines starting at 0.5 arcmin
    beam_model_type = "beta_pol"  # 'bsplines', 'gaussian', 'beta_pol', 'beta_T', or 'bsplines_plus_gaussian'

    # Default bounds for each beam model type
    beam_model_bounds = {
        "bsplines": {"T_coeffs": (-0.5, 1.5), "P_coeffs": (-0.5, 1.5)},
        "gaussian": {"T_width_arcmin": (0.5, 2.0), "P_width_arcmin": (0.5, 2.0)},
        "beta_pol": {"beta_pol": (-0.5, 2.0)},
        "beta_T": {"beta_T": (-0.5, 2.0)},
        "bsplines_plus_gaussian": {
            "gaussian_sigma_arcmin": (0.1, 1.0),
            "bspline_coeffs_T": (-0.5, 1.5),
            "bspline_coeffs_P": (-0.5, 1.5),
        },
    }

    @property
    def beam_coeff_bounds(self):
        """Get bounds for current beam model."""
        return self.beam_model_bounds[self.beam_model_type]

    bsplines_gaussian_rmin_arcmin = 0.5

    skip_sources = [
        "J010644-4034.4",  # extended source
        "J215706-6941.3",  # extended source
        "J210933-4110.4",  # extended source
        "J202724-7007.2",  # extended source
        "J051949-4546.7",  # triple source
        "J051926.340-4545.9",  # that same triple source
        "J051545-4556.7",  # has another bright source in the cutout
        "J050644-6109.6",  # has another bright source in the cutout
        "J011546-3049.3",  # has another bright source in the cutout
        "J005802-3234.3",  # has another bright source in the cutout
        "J133639-3357.9",  # extended source
        "J135839-3958.0",  # near an extended source
        "J135653-4006.2",  # near that same extended source
        "J135747-4006.3",  # near that same extended source
        "J233611-5236.7",  # significantly off-center
        "J103743-2823.2",  # significantly off-center
        "J142756-4206.3",  # source with a weird stripe down the middle
        "J053850-4405.1",  # another one with a weird stripe down the middle
        "J061030-6058.6",  # noisy, no appreciable amount of signal
    ]

    source_param_names = ["yoff", "xoff", "flux"]

    # Band-specific FWHM values, in arcminutes
    band_fwhm_arcmin = {"90GHz": 1.509, "150GHz": 1.108, "220GHz": 0.938}

    # Chi-squared normalization for sampling
    chi2_normalization = 1.0

    # MCMC controls
    mcmc_num_warmup = 1000
    mcmc_num_samples = 1000
    mcmc_target_accept = 0.8
    mcmc_max_tree_depth = 10  # increase to 12–14 if many divergences

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    @property
    def dtype_np_real(self):
        return np.float64 if self.double_precision else np.float32

    @property
    def dtype_np_complex(self):
        return np.complex128 if self.double_precision else np.complex64

    @property
    def dtype_jax_real(self):
        return jnp.float64 if self.double_precision else jnp.float32

    @property
    def dtype_jax_complex(self):
        return jnp.complex128 if self.double_precision else jnp.complex64
