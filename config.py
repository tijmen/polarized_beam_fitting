"""
Configuration module for polarized beam fitting.

Contains all the constants and parameters used in the analysis.
"""

import os
from dataclasses import dataclass

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
    ]
    noise_psd_path = "/home/tijmen/cmb_analysis/beam_analysis/data/subfield_noise_PSD_{band}GHz_mean_sub2.fits"
    output_dir = "/home/tijmen/cmb_analysis/beam_analysis/output"
    cache_dir = "/home/tijmen/cmb_analysis/beam_analysis/cache"

    # Analysis parameters
    bands = ["90GHz"]  # ["90GHz", "150GHz", "220GHz"]  # Frequency bands for analysis

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

    # Noise PSD options
    # Choose noise PSD method from available options:
    # 'clusterfinder_psd': Load pre-computed clusterfinder instrument noise PSD from FITS file
    # 'kx_averaged_individual': Calculate individual noise PSDs using k_x averaging with max heuristic
    # 'white_noise_scaled': White noise assumption rescaled to center-excised standard deviation
    # 'ensemble_asd_mean': Average amplitude spectral densities across sources then convert to PSD
    noise_psd_method = "clusterfinder_psd"

    # Data-driven Noise PSD parameters (used when noise_psd_method = 'data_driven')
    # Calculates PSD directly from regions of the map away from the source
    # We start by creating a modified apodization mask with a configurable
    # hole in the center, then calculate the noise PSD from the empty regions.
    noise_hole_radius_arcmin = 4.0  # Radius of central hole for noise calculation (arcmin)

    # Optimization parameters
    n_steps = 15000
    max_sources = None  # set to a finite number to only calculate the likelihood over a few sources e.g. for testing
    pol_focus = 1.0  # Factor to upweight the importance of polarization likelihood

    # Bootstrap resampling parameters
    enable_bootstrap = False  # Enable/disable bootstrap uncertainty estimation
    n_bootstrap_samples = 100  # Number of bootstrap iterations
    bootstrap_seed = 42  # Random seed for reproducibility (None for random)
    bootstrap_confidence_levels = [68, 95]  # Confidence levels to report (percentiles)
    bootstrap_warm_start = False  # If True, warm-start from previous optimum; if False, use same initial params as global fit

    # Plotting options
    n_diagnostic_plots = 3  # Number of highest chi2 sources to plot diagnostics for.
    # Can be an integer (default 3), "all" to plot all sources,
    # or 0 to disable diagnostic plots entirely

    source_bounds = (
        (-5.0, 5.0),  # yoff (y_offset)
        (-5.0, 5.0),  # xoff (x_offset)
        (0.1, 10.0),  # flux_correction (applies to all T, Q, U)
    )

    # Initial parameter values
    source_inits = (-0.49, -0.49, 0.99)  # yoff, xoff, flux_correction

    # Debug options
    debug = False

    # Select which can of parameterized model you want to fit.
    # The betapol model just has one free parameter
    # The Gaussian model frees the width of the T and P beams separately
    # The B-spline model is very flexible, but suffers from the P beam spike problem described at
    #   https://pole.uchicago.edu/spt3g/index.php/Polarized_Point_Source_Stack#An_Explanation_for_the_Peaky_P_Beam
    # The bsplines_plus_gaussian model combines a central Gaussian with area-normalized B-splines starting at 0.5 arcmin
    beam_model_type = "betapol"  # 'b_spline', 'gaussian', 'betapol', 'betatest', or 'bsplines_plus_gaussian'

    # Default bounds for each beam model type
    beam_model_bounds = {
        "b_spline": {"T_coeffs": (-0.5, 1.5), "P_coeffs": (-0.5, 1.5)},
        "gaussian": {"T_width_arcmin": (0.5, 2.0), "P_width_arcmin": (0.5, 2.0)},
        "betapol": {"beta_pol": (-0.5, 2.0)},
        "betatest": {"beta_T": (-0.5, 2.0)},
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

    source_param_names = ["yoff", "xoff", "flux_correction"]

    # Band-specific FWHM values, measured in arcminutes
    band_fwhm_arcmin = {"90GHz": 1.509, "150GHz": 1.108, "220GHz": 0.938}

    # Chi-squared normalization for sampling
    chi2_normalization = 1.0

    # NUTS / MCMC controls
    nuts_num_warmup = 1000
    nuts_num_samples = 1000
    nuts_num_chains = 1  # set to 2–4 if memory/GPU allows
    nuts_chain_method = "parallel"  # or "sequential" on single-core
    nuts_target_accept = 0.8
    nuts_max_tree_depth = 10  # increase to 12–14 if many divergences
    nuts_dense_mass = False  # can be list of blocks if you later want structure
    nuts_adapt_step_size = True
    nuts_adapt_mass_matrix = True
    nuts_find_heuristic_step_size = False
    nuts_forward_mode = False
    nuts_progress_bar = True
    nuts_thin = 1
    nuts_seed = 0

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
