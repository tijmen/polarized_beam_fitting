"""Module with configuration defaults for polarized beam fitting."""

import os
from typing import Generator, Tuple

import jax.numpy as jnp
import numpy as np
from spt3g import core

from .fields import FieldCatalog


class BeamFittingConfig:
    """Configuration class for polarized beam fitting analysis."""

    # === Filesystem & data sources ===
    output_dir = "/home/tijmen/cmb_analysis/beam_analysis/output"
    cache_dir = "/home/tijmen/cmb_analysis/beam_analysis/cache"
    coadd_filenames = {
        "winter": [
            "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_tau_decon_winter.g3",
        ],
        "summer_a": [
            "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_tau_decon_summera.g3",
        ],
        "summer_b": [
            "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_tau_decon_summerb.g3",
        ],
        "summer_c": [
            "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_tau_decon_summerc.g3",
        ],
        # "winter_nodecon": [
        #     "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_19-20_winter.g3",
        # ],
        # "summer_a_nodecon": [
        #     "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_summera.g3",
        # ],
        # "summer_b_nodecon": [
        #     "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_summerb.g3",
        # ],
        # "summer_c_nodecon": [
        #     "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_summerc.g3",
        # ],
        # "targeted1": ["/home/tijmen/cmb_analysis/beam_analysis/data/J1924-2914.g3"],
        # "targeted2": ["/home/tijmen/cmb_analysis/beam_analysis/data/J2258-2758.g3"],  # these targeted sources are sus
    }
    noise_psd_path = "/home/tijmen/cmb_analysis/beam_analysis/data/subfield_noise_PSD_{band}GHz_mean_sub2.fits"
    betapol_data_path = "/home/tijmen/cmb_analysis/beam_analysis/polarized_beam_fitting/data/betapol_TdH.npz"
    leakage_template_dir = os.path.join(cache_dir, "leakage_templates")

    for this_dir in [output_dir, cache_dir, leakage_template_dir]:
        os.makedirs(this_dir, exist_ok=True)

    # === Run selection & numerics ===
    bands = ["90GHz"]  # ["90GHz", "150GHz", "220GHz"]  # Frequency bands for analysis
    double_precision = True  # Use 64-bit precision for all calculations
    debug = False

    # === Map geometry ===
    map_size_pix = 300
    reso_arcmin = 0.1
    apodization_width_pix = 10

    # === Beam modeling ===
    knot_spacing_arcmin = 0.4
    spline_k = 4  # Cubic B-spline
    spline_rmax_arcmin = 15.0
    beam_model_type = "beta_pol"  # 'gaussian', 'beta_pol', 'beta_T', 'bsplines_plus_gaussian', or 'bsplines_plus_main'
    bsplines_gaussian_rmin_arcmin = 0.5
    bsplines_main_rmin_arcmin = 0.75
    band_fwhm_arcmin = {"90GHz": 1.509, "150GHz": 1.108, "220GHz": 0.938}
    bsplines_gaussian_baseline = {"90GHz": 9e-5, "150GHz": 4e-5, "220GHz": 5e-5}
    beam_model_bounds = {
        "gaussian": {"T_width_arcmin": (0.5, 2.0), "P_width_arcmin": (0.5, 2.0)},
        "beta_pol": {"beta_pol": (-0.5, 2.0)},
        "beta_T": {"beta_T": (-0.5, 2.0)},
        "bsplines_plus_gaussian": {
            "gaussian_sigma_arcmin": (0.1, 1.0),
            "bspline_coeffs_T": (-0.5, 1.5),
            "bspline_coeffs_P": (-0.5, 1.5),
        },
        "bsplines_plus_main": {
            "bspline_coeffs_P": (-0.5, 1.5),
        },
    }

    source_param_names = ["yoff", "xoff", "flux"]
    source_flux_bounds = (-5.0, 100.0)  # T, Q, U flux in mK
    source_position_bounds = (
        (-5.0, 5.0),  # yoff (source center y offset in pixels)
        (-5.0, 5.0),  # xoff (source center x offset in pixels)
    )

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
        "J235935-3133.7",  # too faint at 220
        "J002430-2928.8",  # too faint at 150
        "J032046-3837.4",  # too faint at 150
        "J034838-2749.2",  # second noisiest-source. Don't know why.
        "J045703-2324.8",  # 3x noisier cutout map than others from this field?!
        "J021046-5101.0",  # "focus quasar" Lots of S/N, but non-uniform cov and weird residuals
        "J044017-4333.1",  # also weird stuff in the residual maps
        "J231544-5018.6",  # non-uniform coverage
        "J052257-3627.5",  # Might be slightly extended according to radio imaging
        "J010645-4034.3",  # Alternative ID for J010644-4034.4
        # "J023653-6136.2",  # outlier in the real-space/white-noise comparison, and slightly digs into the map edge at the bottom
    ]

    # === Source selection & leakage handling ===
    min_t_amplitude = 300 * core.G3Units.uK
    max_zero_fraction = 0.05
    leakage_weighting = "linear"  # "flat", "linear", "quadratic", or "median"
    use_precomputed_leakage_templates = True  # Use offline templates stored on disk

    # === CMB calibration factors ===
    # Tcal https://sptlocal.grid.uchicago.edu/~yomori/20192020_lensing/Tcal/v3/spt3g20192020_tcal.html#updated-calibration
    # Pcal https://pole.uchicago.edu/spt3g/index.php/File:20231009_EETETT_Updates.pdf
    cmb_calibration_factors = {
        "T": {"90GHz": 1.07, "150GHz": 1.02, "220GHz": 1.01},
        "Q": {"90GHz": 1.05, "150GHz": 1.06, "220GHz": 1.17},
        "U": {"90GHz": 1.05, "150GHz": 1.06, "220GHz": 1.17},
    }

    # === CDRC deprojection/rotation/calibration ===
    use_cdrc = False  # Enable Wei's CDRC procedure for deprojecting, rotating, and calibrating T/Q/U maps.
    cdrc_winter_params = {
        "ra0hdec-44.75": {
            "90GHz": {"delta_psi": 0.0062, "epsilon_q_tt": 0.0025, "epsilon_u_tt": 0.0054},
            "150GHz": {"delta_psi": 0.0051, "epsilon_q_tt": 0.0026, "epsilon_u_tt": 0.0072},
            "220GHz": {"delta_psi": -0.0113, "epsilon_q_tt": 0.0026, "epsilon_u_tt": 0.0081},
        },
        "ra0hdec-52.25": {
            "90GHz": {"delta_psi": 0.0080, "epsilon_q_tt": 0.0031, "epsilon_u_tt": 0.0060},
            "150GHz": {"delta_psi": 0.0062, "epsilon_q_tt": 0.0030, "epsilon_u_tt": 0.0070},
            "220GHz": {"delta_psi": -0.0122, "epsilon_q_tt": 0.0038, "epsilon_u_tt": 0.0065},
        },
        "ra0hdec-59.75": {
            "90GHz": {"delta_psi": 0.0099, "epsilon_q_tt": 0.0076, "epsilon_u_tt": 0.0089},
            "150GHz": {"delta_psi": 0.0087, "epsilon_q_tt": 0.0094, "epsilon_u_tt": 0.0130},
            "220GHz": {"delta_psi": 0.0016, "epsilon_q_tt": 0.0186, "epsilon_u_tt": 0.0111},
        },
        "ra0hdec-67.25": {
            "90GHz": {"delta_psi": 0.0093, "epsilon_q_tt": 0.0065, "epsilon_u_tt": 0.0087},
            "150GHz": {"delta_psi": 0.0078, "epsilon_q_tt": 0.0088, "epsilon_u_tt": 0.0118},
            "220GHz": {"delta_psi": 0.0060, "epsilon_q_tt": 0.0181, "epsilon_u_tt": 0.0132},
        },
    }

    # === Noise and chi-squared evaluation ===
    chi2_method = "fourier"  # "fourier" or "real_space"

    # === Precision estimation settings ===
    precision_n_pca = 0  # Number of PCA components for modeling per-source noise variation. 0 = simple mean
    precision_model_cmb = True  # Use CAMB to estimate CMB contribution to off-diagonals
    precision_datadriven_offdiagonals = False  # Use data-driven band-band-stokes-stokes off-diagonals
    precision_white_noise = False  # Use simple white noise precision

    noise_hole_radius_arcmin = 4.0  # Radius of central hole for noise calculation (arcmin)
    chi2_normalization = 1.0
    ellmax = 31_000  # Multipole cutoff used when operating in Fourier space. 31000 is below 0.85 Nyquist of TOD for all winter/summer data

    # === Optimization and convergence ===
    solver = "tuned"  # "optimistix_bfgs", "optax_adam", "newton_pcg", "tuned"
    bfgs_kwargs = {"atol": 1e-24, "rtol": 1e-24, "verbose": frozenset({"step_size", "loss"})}
    adam_variant = "adam"  # Optax optimizer name ("adam", "amsgrad", ...)
    adam_kwargs = {"learning_rate": 0.001}
    convergence_criterion = "loss_history"  # "absolute_gtol", "relative_gtol", or "loss_history"
    absolute_gtol = 100.0
    relative_gtol = 1.0 / (3e5)  # Default: 1 part in 300,000
    loss_history_length = 200  # Number of steps without improvement before convergence
    n_steps = 8000

    # === Uncertainty estimation ===
    n_bootstrap_samples = 100  # Number of bootstrap iterations
    bootstrap_seed = 42  # Random seed for reproducibility (None for random)
    bootstrap_confidence_levels = [68, 95]  # Confidence levels to report (percentiles)

    # === Diagnostics and plotting ===
    n_diagnostic_plots = 3  # Number of highest chi2 sources to plot diagnostics for.
    # Can be an integer (default 3), "all" to plot all sources,
    # or 0 to disable diagnostic plots entirely

    # === Sampling backends ===
    nuts_num_warmup = 1000
    nuts_num_samples = 1000
    nuts_target_accept = 0.8
    nuts_max_tree_depth = 10

    mclmc_num_warmup = 2000
    mclmc_num_samples = 2000
    mclmc_desired_energy_var = 0.01

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

    @property
    def active_beam_model_bounds(self):
        return self.beam_model_bounds[self.beam_model_type]

    @property
    def field_catalog(self) -> FieldCatalog:
        """Return a FieldCatalog describing all coadd files keyed by observing field."""
        return FieldCatalog(self.coadd_filenames)

    def iter_coadd_files(self) -> Generator[Tuple[str, str], None, None]:
        """Yield (field, filename) pairs for each configured coadd file."""
        yield from self.field_catalog.iter_field_paths()

    @property
    def coadd_file_list(self) -> Tuple[str, ...]:
        """Return a flattened tuple of all coadd file paths."""
        return self.field_catalog.all_paths
