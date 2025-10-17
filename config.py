"""Module with configuration defaults for polarized beam fitting."""

import os
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from spt3g import core


@dataclass
class BeamFittingConfig:
    """Configuration class for polarized beam fitting analysis."""

    # === Filesystem & data sources ===
    output_dir = "/home/tijmen/cmb_analysis/beam_analysis/output"
    cache_dir = "/home/tijmen/cmb_analysis/beam_analysis/cache"
    coadd_filenames = [
        "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_tau_decon_winter.g3",
        "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_tau_decon_summera.g3",
        "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_tau_decon_summerb.g3",
        "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_tau_decon_summerc.g3",
        # "/home/tijmen/cmb_analysis/beam_analysis/data/J1924-2914.g3",
        # "/home/tijmen/cmb_analysis/beam_analysis/data/J2258-2758.g3",  # these targeted sources are sus
    ]
    noise_psd_path = "/home/tijmen/cmb_analysis/beam_analysis/data/subfield_noise_PSD_{band}GHz_mean_sub2.fits"
    betapol_data_path = "/home/tijmen/cmb_analysis/beam_analysis/polarized_beam_fitting/data/betapol_TdH.npz"
    leakage_template_dir = os.path.join(output_dir, "leakage_templates")

    # Ensure output directory exists so downstream consumers can write immediately.
    os.makedirs(output_dir, exist_ok=True)

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
    spline_rmax_arcmin = 10.0
    beam_model_type = "beta_pol"  # 'bsplines', 'gaussian', 'beta_pol', 'beta_T', or 'bsplines_plus_gaussian'
    bsplines_gaussian_rmin_arcmin = 0.5
    bsplines_plus_gaussian_semilogy = False
    band_fwhm_arcmin = {"90GHz": 1.509, "150GHz": 1.108, "220GHz": 0.938}
    beam_model_bounds = {
        "bsplines": {"T_coeffs": (-0.5, 1.5), "P_coeffs": (-0.5, 1.5)},
        "gaussian": {"T_width_arcmin": (0.5, 2.0), "P_width_arcmin": (0.5, 2.0)},
        "beta_pol": {"beta_pol": (-0.5, 2.0)},
        "beta_T": {"beta_T": (-0.5, 2.0)},
        "bsplines_plus_gaussian_linear": {
            "gaussian_sigma_arcmin": (0.1, 1.0),
            "bspline_coeffs_T": (-0.5, 1.5),
            "bspline_coeffs_P": (-0.5, 1.5),
        },
        "bsplines_plus_gaussian_semilogy": {
            "gaussian_sigma_arcmin": (0.1, 1.0),
            "bspline_coeffs_T": (np.log(1e-6), np.log(1.5)),
            "bspline_coeffs_P": (np.log(1e-6), np.log(1.5)),
        },
    }

    source_param_names = ["yoff", "xoff", "flux"]
    source_flux_bounds = (-5.0, 100.0)  # T, Q, U flux in uK
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
        "J045703-2324.8",  # noisiest source by a factor of 3! Definitely don't know why.
    ]

    # === Source selection & leakage handling ===
    min_t_amplitude = 300 * core.G3Units.uK
    max_zero_fraction = 0.05
    exclude_from_leakage_template = None
    leakage_weighting = "linear"  # "flat", "linear", "quadratic", or "median"
    use_precomputed_leakage_templates = True  # Use offline templates stored on disk

    # === Noise and chi-squared evaluation ===
    chi2_method = "fourier"  # "fourier" or "real_space"
    # noise_psd_method options:
    #   "clusterfinder_psd"          : load fixed PSD from FITS (single-band only).
    #   "kx_averaged"                : 1D PSD averaged along kx, diagonal in (band, stokes).
    #   "ensemble_asd_mean"          : take ASD per source, average, square back to PSD per band/stokes.
    #   "white_noise"                : unit PSD with 2× scaling for Q/U (simple test case).
    #   "multiband_covariance"       : full (band × stokes) covariance estimated from source ensemble.
    #   "pca_multiband_covariance"   : PCA-regularized multiband covariance (complex precision output).
    #   "parametric_precision"       : CAMB + analytic noise model yielding dense precision matrices.
    #   "pca_psd_separate_tqu"       : log-space PCA per stokes, diagonal in band; default.
    noise_psd_method = "pca_psd_separate_tqu"
    n_pca_components = 4
    noise_hole_radius_arcmin = 4.0  # Radius of central hole for noise calculation (arcmin)
    chi2_normalization = 1.0
    ellmax = 31_000  # Multipole cutoff used when operating in Fourier space. 31000 is below 0.85 Nyquist of TOD for all winter/summer data

    # === Optimization and convergence ===
    solver = "optax_adam"  # "optimistix_bfgs", "optax_adam", "newton_pcg"
    bfgs_kwargs = {"atol": 1e-24, "rtol": 1e-24, "verbose": frozenset({"step_size", "loss"})}
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
    mclmc_desired_energy_var = 5e-4

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
