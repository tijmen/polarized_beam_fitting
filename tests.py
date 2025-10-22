"""End-to-end and unit tests for the polarized beam fitting package."""

import os
import tempfile
import unittest
from unittest.mock import patch

import camb
import healpy as hp
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from astropy.io import fits

matplotlib.use("Agg")

from polarized_beam_fitting.beam_model import create_beam_model
from polarized_beam_fitting.config import BeamFittingConfig
from polarized_beam_fitting.data_loader import DataLoader
from polarized_beam_fitting.fitter import PolarizedBeamFitter
from polarized_beam_fitting.plotting import BeamPlotter, create_diagnostic_plots
from polarized_beam_fitting.precision import (
    _compute_cmb_covariance,
    _smooth_highpass_1d,
)
from polarized_beam_fitting.utils import (
    calculate_tod_nyquist_radial_mask_smooth,
    compute_rectangular_ell_cut_indices,
    ell_grid,
    linear_interp_differentiable,
    make_apodization_mask,
    parse_declination,
    predict_nyquist_ell_x,
    safe_filename,
    shift_map_bilinear,
)


def get_test_config(**kwargs):
    """
    Creates a test config based on the defaults in BeamFittingConfig,
    with overrides for fast, isolated testing and any specified kwargs.
    """
    config = BeamFittingConfig()

    # --- Default overrides for testing ---
    config.map_size_pix = 64
    config.reso_arcmin = 0.2
    config.apodization_width_pix = 8
    config.n_steps = 20000
    config.bands = ["150GHz"]
    config.covariance_method = "white_noise"
    config.min_t_amplitude = 0
    config.cache_dir = tempfile.mkdtemp()
    config.coadd_filenames = {"test_field": ["mock_file.g3"]}
    config.n_diagnostic_plots = 0
    config.bfgs_kwargs = {"atol": 1e-24, "rtol": 1e-24, "verbose": frozenset({})}

    # --- Apply user-specified overrides ---
    for key, value in kwargs.items():
        setattr(config, key, value)

    # --- Handle special cases based on config ---
    if config.beam_model_type == "beta_pol":
        dummy_betapol_file = os.path.join(config.cache_dir, "betapol_tests.npz")
        r_fine = np.linspace(0, 10, 100)
        sigma_main, sigma_bt = 1.2 / 2.355, 1.4 / 2.355
        bmain = np.exp(-0.5 * (r_fine / sigma_main) ** 2)
        bt = np.exp(-0.5 * (r_fine / sigma_bt) ** 2)
        betapol_payload = {"r_fine_arcmin": r_fine}
        for band in config.bands:
            suffix = band.replace("GHz", "")
            betapol_payload[f"BT_r_norm_{suffix}"] = bt
            betapol_payload[f"Bmain_r_norm_{suffix}"] = bmain
        np.savez(dummy_betapol_file, **betapol_payload)
        config.betapol_data_path = dummy_betapol_file

    elif config.beam_model_type == "bsplines_plus_gaussian":
        dummy_betapol_file = os.path.join(config.cache_dir, "betapol_tests.npz")
        r_fine = np.linspace(0, 10, 100)
        sigma = config.band_fwhm_arcmin[config.bands[0]] / (2 * np.sqrt(2 * np.log(2)))
        beam_profile = np.exp(-0.5 * (r_fine / sigma) ** 2)
        betapol_payload = {"r_fine_arcmin": r_fine}
        for band in config.bands:
            suffix = band.replace("GHz", "")
            betapol_payload[f"BT_r_norm_{suffix}"] = beam_profile
            betapol_payload[f"Bmain_r_norm_{suffix}"] = beam_profile
        np.savez(dummy_betapol_file, **betapol_payload)
        config.betapol_data_path = dummy_betapol_file

    if config.covariance_method == "clusterfinder_psd":
        psd_file = tempfile.NamedTemporaryFile(delete=False, suffix=".fits").name
        psd_data = np.ones((256, 256), dtype=config.dtype_np_real)
        fits.writeto(psd_file, psd_data, overwrite=True)
        config.noise_psd_path = psd_file

    return config


def _normalize_true_params(config, true_beam_params):
    """Return per-band parameter dictionaries for synthetic data generation."""
    n_bands = len(config.bands)
    if isinstance(true_beam_params, (list, tuple)):
        if len(true_beam_params) != n_bands:
            raise ValueError("Length of true_beam_params list must match number of bands.")
        return list(true_beam_params)
    if isinstance(true_beam_params, dict):
        return [dict(true_beam_params) for _ in range(n_bands)]
    raise TypeError("true_beam_params must be a dict or list of dicts.")


def generate_mock_data(config, true_beam_params, n_sources=3):
    """
    Generate mock data based on a specific beam model and true parameters.
    """
    np.random.seed(42)
    shape = (config.map_size_pix, config.map_size_pix)
    ny_full, nx_full = shape
    y_grid, x_grid = np.ogrid[-ny_full // 2 : ny_full // 2, -nx_full // 2 : nx_full // 2]

    band_params = _normalize_true_params(config, true_beam_params)
    n_bands = len(config.bands)
    beam_models = [create_beam_model(config, y_grid, x_grid, band) for band in config.bands]

    true_yoffs = np.random.uniform(-1.5, 1.5, n_sources)
    true_xoffs = np.random.uniform(-1.5, 1.5, n_sources)
    amp_low = np.array([0.9, -0.05, -0.05])
    amp_high = np.array([1.1, 0.05, 0.05])
    true_amps = amp_low + (amp_high - amp_low) * np.random.uniform(size=(n_sources, n_bands, 3))

    maps_numpy = np.zeros((n_sources, ny_full, nx_full, n_bands, 3), dtype=config.dtype_np_real)
    weights_numpy = np.zeros((n_sources, ny_full, nx_full, n_bands, 3, 3), dtype=config.dtype_np_real)

    for i in range(n_sources):
        for band_idx, beam_model in enumerate(beam_models):
            true_beam_T, true_beam_P = beam_model.evaluate_beam_maps(band_params[band_idx], true_yoffs[i], true_xoffs[i])
            signal_maps = np.stack(
                [
                    true_amps[i, band_idx, 0] * true_beam_T,
                    true_amps[i, band_idx, 1] * true_beam_P,
                    true_amps[i, band_idx, 2] * true_beam_P,
                ],
                axis=-1,
            )

            noise_level = 1e-5
            noise_maps = np.stack(
                [
                    np.random.normal(0, noise_level, shape),
                    np.random.normal(0, noise_level * np.sqrt(2), shape),
                    np.random.normal(0, noise_level * np.sqrt(2), shape),
                ],
                axis=-1,
            )

            maps_numpy[i, :, :, band_idx, :] = signal_maps + noise_maps
            weights_numpy[i, :, :, band_idx, 0, 0] = 1.0 / (noise_level**2)
            weights_numpy[i, :, :, band_idx, 1, 1] = 1.0 / ((noise_level * np.sqrt(2)) ** 2)
            weights_numpy[i, :, :, band_idx, 2, 2] = 1.0 / ((noise_level * np.sqrt(2)) ** 2)

    apod_mask = make_apodization_mask(shape, config.apodization_width_pix)
    maps_apodized = maps_numpy * apod_mask[np.newaxis, :, :, np.newaxis, np.newaxis]
    maps_fft_numpy = np.fft.fft2(maps_apodized, axes=(1, 2))

    gaussfit_yoff = np.tile(true_yoffs[:, None], (1, n_bands))
    gaussfit_xoff = np.tile(true_xoffs[:, None], (1, n_bands))
    gaussfit_initial_amp = true_amps

    raw_maps = np.zeros_like(maps_numpy)
    qu_templates = np.zeros((ny_full, nx_full, n_bands, 2))
    source_ids = np.array([f"mock_source_{i}" for i in range(n_sources)])
    source_fields = np.array(["mock_field" for _ in range(n_sources)], dtype=object)

    if config.chi2_method == "fourier":
        # Apply the same ell truncation as the real data loader
        idx_y, idx_x = compute_rectangular_ell_cut_indices((ny_full, nx_full), config.reso_arcmin, config.ellmax)
        if idx_y is not None and idx_x is not None:
            ny, nx = len(idx_y), len(idx_x)
            # Truncate maps to match precision matrix size
            maps_numpy = maps_numpy[:, idx_y, :, :, :][:, :, idx_x, :, :]
            maps_fft_numpy = maps_fft_numpy[:, idx_y, :, :, :][:, :, idx_x, :, :]
            weights_numpy = weights_numpy[:, idx_y, :, :, :, :][:, :, idx_x, :, :, :]
            qu_templates = qu_templates[idx_y, :, :, :][:, idx_x, :, :]
            precision_placeholder = np.ones((n_sources, ny, nx, n_bands, 3, n_bands, 3), dtype=config.dtype_np_real)
        else:
            precision_placeholder = np.ones((n_sources, ny_full, nx_full, n_bands, 3, n_bands, 3), dtype=config.dtype_np_real)
        debug_placeholder = None
    else:
        precision_placeholder = None
        debug_placeholder = None

    return (
        gaussfit_yoff,
        gaussfit_xoff,
        gaussfit_initial_amp,
        raw_maps,
        qu_templates,
        maps_numpy,
        weights_numpy,
        maps_fft_numpy,
        source_ids,
        source_fields,
        n_sources,
        precision_placeholder,
        debug_placeholder,
    )


class TestTemplateHandling(unittest.TestCase):
    """Unit tests for handling of precomputed templates."""

    def test_precomputed_template_shift(self):
        config = get_test_config()
        config.use_precomputed_leakage_templates = True
        ny = nx = 7
        config.map_size_pix = ny
        config.apodization_width_pix = 2

        loader = DataLoader(config)
        n_src = 1
        n_bands = len(config.bands)

        gaussfit_amp = np.zeros((n_src, n_bands, 3), dtype=config.dtype_np_real)
        gaussfit_amp[0, 0, 0] = 2.0

        raw_maps = np.zeros((n_src, ny, nx, n_bands, 3), dtype=config.dtype_np_real)
        raw_maps[0, :, :, 0, 1] = 5.0
        raw_maps[0, :, :, 0, 2] = -4.0

        template = np.zeros((ny, nx, n_bands, 2), dtype=config.dtype_np_real)
        template[ny // 2, nx // 2, 0, 0] = 1.0
        template[ny // 2, nx // 2, 0, 1] = 0.3
        qu_templates = {"mock_field": template}

        source_fields = np.array(["mock_field"], dtype=object)
        source_ids = np.array(["mock_source"])
        gaussfit_yoff = np.array([[0.4]], dtype=config.dtype_np_real)
        gaussfit_xoff = np.array([[-0.6]], dtype=config.dtype_np_real)

        maps_clean, _ = loader._prepare_clean_maps(
            gaussfit_amp,
            raw_maps,
            qu_templates,
            source_fields,
            source_ids,
            gaussfit_yoff,
            gaussfit_xoff,
            template_flux=gaussfit_amp,
        )

        shifted_q = shift_map_bilinear(template[:, :, 0, 0], gaussfit_yoff[0, 0], gaussfit_xoff[0, 0])
        shifted_u = shift_map_bilinear(template[:, :, 0, 1], gaussfit_yoff[0, 0], gaussfit_xoff[0, 0])

        expected_q = raw_maps[0, :, :, 0, 1] - shifted_q * gaussfit_amp[0, 0, 0]
        expected_u = raw_maps[0, :, :, 0, 2] - shifted_u * gaussfit_amp[0, 0, 0]

        np.testing.assert_allclose(maps_clean[0, :, :, 0, 1], expected_q, rtol=0, atol=1e-6)
        np.testing.assert_allclose(maps_clean[0, :, :, 0, 2], expected_u, rtol=0, atol=1e-6)


@patch("polarized_beam_fitting.fitter.DataLoader")
class TestEndToEndRecovery(unittest.TestCase):
    """
    End-to-end tests verifying parameter recovery.
    """

    def run_test_and_assert(
        self,
        mock_data_loader,
        config,
        true_params,
        assertion_func,
    ):
        if config.double_precision:
            jax.config.update("jax_enable_x64", True)
        else:
            jax.config.update("jax_enable_x64", False)

        mock_data = generate_mock_data(config, true_params)
        mock_data_loader.return_value.load_and_prepare.return_value = mock_data

        fitter = PolarizedBeamFitter(config=config)
        best_fit_params = fitter.run_fit()

        print(f"true: {true_params}")
        print(f"best_fit: {best_fit_params['beams'][0]}")
        assertion_func(best_fit_params["beams"], true_params)
        return fitter, best_fit_params

    def _assert_gaussian_bspline_recovery(self, config, recovered_params, true_params):
        """Check Gaussian+Bspline recovery by comparing sigma and coefficient norms."""
        sigma_fit = float(np.asarray(recovered_params["gaussian_sigma_arcmin"]))
        sigma_true = float(true_params["gaussian_sigma_arcmin"])
        self.assertAlmostEqual(sigma_fit, sigma_true, delta=1e-3)

        fit_T = np.asarray(recovered_params["bspline_coeffs_T"])
        fit_P = np.asarray(recovered_params["bspline_coeffs_P"])
        true_T = np.asarray(true_params["bspline_coeffs_T"])
        true_P = np.asarray(true_params["bspline_coeffs_P"])

        self.assertLess(np.linalg.norm(fit_T - true_T), 0.1)
        self.assertLess(np.linalg.norm(fit_P - true_P), 0.1)

    def test_default_config(self, *mocks):
        print("\n--- Testing Default Configuration (betapol, fourier, double precision) ---")
        config = get_test_config()
        true_params = {"beta_pol": 0.75}
        self.run_test_and_assert(
            *mocks,
            config,
            true_params,
            lambda fit, true: self.assertAlmostEqual(fit[0]["beta_pol"], true["beta_pol"], delta=0.01),
        )
        print("✓ Default config test successful.")

    def test_beam_model_gaussian(self, *mocks):
        print("\n--- Testing Beam Model: Gaussian ---")
        config = get_test_config(beam_model_type="gaussian")
        true_fwhm = config.band_fwhm_arcmin[config.bands[0]]
        true_params = {"T_width_arcmin": true_fwhm, "P_width_arcmin": true_fwhm}
        self.run_test_and_assert(
            *mocks,
            config,
            true_params,
            lambda fit, true: self.assertAlmostEqual(fit[0]["T_width_arcmin"], true["T_width_arcmin"], delta=0.01),
        )
        print("✓ Gaussian model test successful.")

    def test_beam_model_bspline_plus_gaussian(self, *mocks):
        print("\n--- Testing Beam Model: Gaussian + B-splines ---")
        config = get_test_config(beam_model_type="bsplines_plus_gaussian")
        ny = nx = config.map_size_pix
        y, x = np.ogrid[-ny // 2 : ny // 2, -nx // 2 : nx // 2]
        beam_model = create_beam_model(config, y, x, config.bands[0])
        gaussian_sigma = config.band_fwhm_arcmin[config.bands[0]] / (2 * np.sqrt(2 * np.log(2)))
        zero_coeffs = np.zeros(beam_model.n_bspline_coeffs, dtype=config.dtype_np_real)
        true_params = {
            "gaussian_sigma_arcmin": gaussian_sigma,
            "bspline_coeffs_T": zero_coeffs,
            "bspline_coeffs_P": zero_coeffs,
        }
        self.run_test_and_assert(
            *mocks,
            config,
            true_params,
            lambda fit, true: self._assert_gaussian_bspline_recovery(config, fit[0], true),
        )
        print("✓ Gaussian + B-splines model test successful.")

    def test_chi2_method_real_space(self, *mocks):
        print("\n--- Testing Chi2 Method: Real Space ---")
        config = get_test_config(chi2_method="real_space", n_steps=150000)
        true_params = {"beta_pol": 0.75}
        self.run_test_and_assert(
            *mocks,
            config,
            true_params,
            lambda fit, true: self.assertAlmostEqual(fit[0]["beta_pol"], true["beta_pol"], delta=0.01),
        )
        print("✓ Real space test successful.")

    def test_single_precision(self, *mocks):
        print("\n--- Testing Precision: Single ---")
        config = get_test_config(double_precision=False)
        true_params = {"beta_pol": 0.75}
        self.run_test_and_assert(
            *mocks,
            config,
            true_params,
            lambda fit, true: self.assertAlmostEqual(fit[0]["beta_pol"], true["beta_pol"], delta=0.05),
        )
        print("✓ Single precision test successful.")

    def test_covariance_kx_averaged(self, *mocks):
        print("\n--- Testing covariance method: kx_averaged ---")
        config = get_test_config(covariance_method="kx_averaged")
        true_params = {"beta_pol": 0.75}
        self.run_test_and_assert(
            *mocks,
            config,
            true_params,
            lambda fit, true: self.assertAlmostEqual(fit[0]["beta_pol"], true["beta_pol"], delta=0.01),
        )
        print("✓ kx_averaged covariance test successful.")

    def test_solver_bfgs(self, *mocks):
        print("\n--- Testing Solver: Optimistix BFGS ---")
        config = get_test_config(solver="optimistix_bfgs", n_steps=300)
        config.bfgs_kwargs = {"atol": 1e-18, "rtol": 1e-18, "verbose": frozenset({})}
        true_params = {"beta_pol": 0.75}
        self.run_test_and_assert(
            *mocks,
            config,
            true_params,
            lambda fit, true: self.assertAlmostEqual(fit[0]["beta_pol"], true["beta_pol"], delta=0.02),
        )
        print("✓ BFGS solver test successful.")

    def test_rebin_psd_with_averaging(self):
        print("\n--- Unit Testing: _rebin_psd_with_averaging ---")
        calc = ClusterfinderPSDCalculator(BeamFittingConfig(), (64, 64))
        psd_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        expected = np.array([[3.5, 5.5], [11.5, 13.5]])
        rebinned = calc._rebin_psd_with_averaging(psd_array, (2, 2))
        np.testing.assert_allclose(rebinned, expected)
        print("✓ Rebinning test successful.")

    def test_cmb_pca_regularizes_diagonals(self):
        print("\n--- Unit Testing: cmb_pca diagonal regularization ---")
        config = get_test_config(
            map_size_pix=4,
            bands=["90GHz"],
            covariance_method="cmb_pca",
        )
        config.n_pca_components = 1

        n_src = 4
        ny = 1
        nx = 3
        n_bands = len(config.bands)
        n_stokes = 3

        diag_values = np.array(
            [
                [0.01, 0.05, 0.05],
                [0.02, 0.06, 0.06],
                [0.30, 0.70, 0.70],
                [0.40, 0.80, 0.80],
            ],
            dtype=config.dtype_np_real,
        )

        covariance = np.zeros((n_src, ny, nx, n_bands, n_stokes, n_bands, n_stokes), dtype=config.dtype_np_real)
        for src in range(n_src):
            covariance[src, 0, :, 0, 0, 0, 0] = diag_values[src]
            covariance[src, 0, :, 0, 1, 0, 0] = 0.25
            covariance[src, 0, :, 0, 0, 0, 1] = 0.25

        ell_x_grid = np.array([[0.0, 4000.0, 6000.0]], dtype=config.dtype_np_real)
        ell_y_grid = np.array([[3100.0, 3100.0, 3100.0]], dtype=config.dtype_np_real)
        np.sqrt(ell_x_grid**2 + ell_y_grid**2)

        calc = CmbPcaCalculator(config, (config.map_size_pix, config.map_size_pix))
        calc.set_source_fields(np.array(["A", "A", "B", "B"], dtype=object))

        diag_inputs = np.zeros((n_src, ny, nx, n_bands, n_stokes), dtype=config.dtype_np_real)
        for src in range(n_src):
            diag_inputs[src, 0, :, 0, 0] = diag_values[src]

        diag_regularized = calc._regularize_diagonals(diag_inputs.copy(), ell_x_grid, ell_y_grid)
        floors = calc._compute_white_noise_floors(diag_inputs, ell_x_grid, ell_y_grid)
        diag_bounded = calc._apply_white_noise_floors(diag_regularized.copy(), floors)

        for band in range(n_bands):
            for stokes in range(n_stokes):
                self.assertTrue(np.all(diag_bounded[:, :, :, band, stokes] >= floors[band, stokes] - 1e-6))

        residual = covariance.copy()
        for band in range(n_bands):
            for stokes in range(n_stokes):
                residual[:, :, :, band, stokes, band, stokes] = diag_bounded[:, :, :, band, stokes]

        diag_result = residual[:, 0, :, 0, 0, 0, 0]
        self.assertTrue(np.all(diag_result >= floors[0, 0] - 1e-6))
        offdiag_result = residual[0, 0, 1, 0, 0, 0, 1]
        self.assertAlmostEqual(offdiag_result, 0.25)
        print("✓ cmb_pca enforces white-noise floors and preserves off-diagonals.")

    def test_cmb_pca_debug_outputs(self):
        print("\n--- Unit Testing: cmb_pca debug payload ---")
        config = get_test_config(
            map_size_pix=4,
            bands=["90GHz"],
            covariance_method="cmb_pca",
        )
        config.n_pca_components = 1
        config.debug = True

        calc = CmbPcaCalculator(config, (config.map_size_pix, config.map_size_pix))
        calc.set_source_fields(np.array(["A", "B"], dtype=object))

        rng = np.random.default_rng(0)
        maps_clean = rng.normal(size=(2, config.map_size_pix, config.map_size_pix, 1, 3)).astype(config.dtype_np_real)
        idx_y = np.array([1, 2])
        idx_x = np.array([0, 3])

        precision, debug = calc.calculate_precision(maps_clean, idx_y=idx_y, idx_x=idx_x)

        self.assertEqual(precision.shape, (2, len(idx_y), len(idx_x), 1, 3, 1, 3))
        self.assertIsNotNone(debug)
        for key in ("covariance_cmb", "covariance_residual", "covariance_regularized"):
            self.assertIn(key, debug)
        print("✓ cmb_pca returns precision and debug covariances when debug is enabled.")


class TestCmbCovarianceNormalization(unittest.TestCase):
    """Validate the CMB covariance normalization against full-sky simulations."""

    def test_flat_sky_periodograms_match_model_covariance(self):
        print("\n--- Unit Testing: CMB covariance normalization ---")
        config = get_test_config(
            map_size_pix=300,
            reso_arcmin=0.1,
            apodization_width_pix=10,
            bands=["150GHz"],
        )
        config.double_precision = True

        ny = nx = config.map_size_pix
        config.reso_arcmin * (np.pi / (180.0 * 60.0))
        reso_deg = config.reso_arcmin / 60.0

        apod_mask = make_apodization_mask((ny, nx), config.apodization_width_pix).astype(config.dtype_np_real)
        _, _, ell_y_grid, ell_x_grid = ell_grid((ny, nx), config.reso_arcmin)
        ell_radial = np.sqrt(ell_x_grid**2 + ell_y_grid**2)

        cov_cmb = _compute_cmb_covariance(config, ny, nx)
        expected_tt = cov_cmb[:, :, 0, 0, 0, 0]

        pars = camb.CAMBparams()
        pars.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.1200, mnu=0.06, omk=0.0, tau=0.0544)
        pars.InitPower.set_params(As=2.1e-9, ns=0.9649)
        ell_max_cmb = 5000
        pars.set_for_lmax(ell_max_cmb, lens_potential_accuracy=0)
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)
        ell_camb = np.arange(powers["total"].shape[0])
        cl_tt = powers["total"][:, 0] * 1e-6  # µK^2 -> mK^2
        cl_tt *= _smooth_highpass_1d(ell_camb, 360.0, 720.0)
        cl_tt = cl_tt[: ell_max_cmb + 1]

        np.random.seed(42)
        nside = 2048
        full_sky_t = hp.synfast(
            cl_tt,
            nside=nside,
            lmax=ell_max_cmb,
            new=True,
            pixwin=False,
            verbose=False,
        )

        pixel_idx = np.arange(ny, dtype=np.float64)
        offsets_deg = (pixel_idx - ny / 2.0 + 0.5) * reso_deg
        theta_base = np.deg2rad(90.0 - offsets_deg)[:, None]
        phi_offsets = np.deg2rad((np.arange(nx, dtype=np.float64) - nx / 2.0 + 0.5) * reso_deg)[None, :]
        theta_grid = np.broadcast_to(theta_base, (ny, nx))

        n_patches = 100
        maps_numpy = np.zeros((n_patches, ny, nx, 1, 3), dtype=config.dtype_np_real)
        for patch_idx in range(n_patches):
            phi_center_rad = np.deg2rad(patch_idx * 0.5)
            phi_grid = np.mod(phi_center_rad + phi_offsets, 2.0 * np.pi)
            patch = hp.get_interp_val(full_sky_t, theta_grid, phi_grid).astype(config.dtype_np_real)
            maps_numpy[patch_idx, :, :, 0, 0] = patch

        covariance, _, _, _, _, _ = _compute_covariance(
            maps_numpy,
            config,
            apod_mask,
            idx_y=None,
            idx_x=None,
        )

        avg = covariance.mean(axis=0)[..., 0, 0, 0, 0]
        std = covariance.std(axis=0, ddof=1)[..., 0, 0, 0, 0]
        stderr = std / np.sqrt(n_patches)

        residual = avg - expected_tt
        comparison_mask = (expected_tt > 1e-9) & (ell_radial <= ell_max_cmb)

        if not np.any(comparison_mask):
            self.fail("No Fourier bins exceeded the significance threshold for CMB covariance validation.")

        normalized_residual = np.zeros_like(residual)
        normalized_residual[comparison_mask] = residual[comparison_mask] / (stderr[comparison_mask] + 1e-12)

        relative_error = np.abs(residual[comparison_mask]) / (np.abs(expected_tt[comparison_mask]) + 1e-12)

        self.assertLess(
            np.nanmax(np.abs(normalized_residual[comparison_mask])),
            5.0,
            "CMB covariance model disagrees with simulations by more than 5σ in some Fourier bins.",
        )
        self.assertLess(
            np.nanmedian(relative_error),
            0.1,
            "Median relative error between simulated periodograms and model covariance exceeds 10%.",
        )
        self.assertLess(
            np.nanpercentile(relative_error, 95),
            0.25,
            "Model covariance differs from simulations by more than 25% in the bulk of Fourier space.",
        )
        print("✓ CMB covariance normalization agrees with simulated flat-sky patches.")


class TestUtilsFunctions(unittest.TestCase):
    """Unit tests for helper utilities."""

    def test_parse_declination_and_nyquist(self):
        decl = parse_declination("J123456-1234.5")
        self.assertAlmostEqual(decl, -12 - 34.5 / 60.0)
        ell_x = predict_nyquist_ell_x(decl)
        self.assertTrue(np.isfinite(ell_x))

    def test_calculate_tod_mask_smooth(self):
        config = get_test_config(map_size_pix=8)
        mask = calculate_tod_nyquist_radial_mask_smooth("J123456-1234.5", (8, 8), config)
        self.assertEqual(mask.shape, (8, 8))
        self.assertGreater(mask.mean(), 0.0)

    def test_linear_interp_differentiable_gradients(self):
        config = get_test_config()
        xp = jnp.linspace(0.0, 1.0, 5)
        fp = xp**2
        x_query = jnp.array([0.25, 0.75])

        values = linear_interp_differentiable(x_query, xp, fp, config)
        np.testing.assert_allclose(np.array(values), np.array([0.0625, 0.5625]), atol=1e-6)

        grad_fn = jax.grad(lambda z: jnp.sum(linear_interp_differentiable(z, xp, fp, config)))
        grads = grad_fn(x_query)
        self.assertTrue(np.all(np.isfinite(np.array(grads))))

    def test_safe_filename(self):
        self.assertEqual(safe_filename("J123456-1234.5"), "J123456_1234_5")


@patch("polarized_beam_fitting.fitter.DataLoader")
class TestPlottingOutputs(unittest.TestCase):
    """Exercise plotting helpers on tiny synthetic fits."""

    def test_create_diagnostic_plots(self, mock_data_loader):
        config = get_test_config(map_size_pix=12, n_steps=800)
        config.n_diagnostic_plots = 1
        config.skip_sources = []

        with tempfile.TemporaryDirectory() as temp_root:
            config.output_dir = os.path.join(temp_root, "default_plots")

            mock_data_loader.return_value.load_and_prepare.return_value = generate_mock_data(
                config,
                {"beta_pol": 0.72},
                n_sources=2,
            )

            fitter = PolarizedBeamFitter(config=config)
            best_fit_params = fitter.run_fit()

            plot_dir = os.path.join(temp_root, "diagnostic_plots")
            os.makedirs(plot_dir, exist_ok=True)
            results = create_diagnostic_plots(fitter, best_fit_params, output_dir=plot_dir)

            for key, value in results.items():
                if not value:
                    continue
                if isinstance(value, list):
                    for filename in value:
                        self.assertTrue(os.path.exists(filename))
                else:
                    self.assertTrue(os.path.exists(value))

            plotter = BeamPlotter(fitter, output_dir=plot_dir)
            profile_path = plotter.plot_beam_profiles(best_fit_params)
            if isinstance(profile_path, list):
                for path in profile_path:
                    self.assertTrue(os.path.exists(path))
            else:
                self.assertTrue(os.path.exists(profile_path))

        print("✓ Plotting pipeline produces files in temporary directory.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
