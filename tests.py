"""End-to-end and unit tests for the polarized beam fitting package."""

import os
import tempfile
import unittest
from unittest.mock import patch

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
from polarized_beam_fitting.noise_psd import (
    ClusterfinderPSDCalculator,
    EnsembleAsdMeanCalculator,
    MultiBandCovarianceCalculator,
    ParametricPrecisionCalculator,
    PcaMultiBandCalculator,
)
from polarized_beam_fitting.plotting import BeamPlotter, create_diagnostic_plots
from polarized_beam_fitting.utils import (
    calculate_tod_nyquist_radial_mask_smooth,
    compute_2d_asd,
    linear_interp_differentiable,
    make_apod_mask_center_excised,
    make_apodization_mask,
    parse_declination,
    predict_nyquist_kx,
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
    config.noise_psd_method = "white_noise"
    config.min_t_amplitude = 0
    config.cache_dir = tempfile.mkdtemp()
    config.coadd_filenames = ["mock_file.g3"]
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

    if config.noise_psd_method == "clusterfinder_psd":
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


def generate_mock_data(config, true_beam_params, n_sources=3, noise_payload=None):
    """
    Generate mock data based on a specific beam model and true parameters.
    """
    np.random.seed(42)
    shape = (config.map_size_pix, config.map_size_pix)
    ny, nx = shape
    y_grid, x_grid = np.ogrid[-ny // 2 : ny // 2, -nx // 2 : nx // 2]

    band_params = _normalize_true_params(config, true_beam_params)
    n_bands = len(config.bands)
    beam_models = [create_beam_model(config, y_grid, x_grid, band) for band in config.bands]

    true_yoffs = np.random.uniform(-1.5, 1.5, n_sources)
    true_xoffs = np.random.uniform(-1.5, 1.5, n_sources)
    amp_low = np.array([0.9, -0.05, -0.05])
    amp_high = np.array([1.1, 0.05, 0.05])
    true_amps = amp_low + (amp_high - amp_low) * np.random.uniform(size=(n_sources, n_bands, 3))

    maps_numpy = np.zeros((n_sources, ny, nx, n_bands, 3), dtype=config.dtype_np_real)
    weights_numpy = np.zeros((n_sources, ny, nx, n_bands, 3, 3), dtype=config.dtype_np_real)

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

    gaussfit_yoff = true_yoffs
    gaussfit_xoff = true_xoffs
    gaussfit_initial_amp = true_amps

    raw_maps = np.zeros_like(maps_numpy)
    qu_templates = np.zeros((ny, nx, n_bands, 2))
    source_ids = np.array([f"mock_source_{i}" for i in range(n_sources)])
    source_fields = np.array(["mock_field" for _ in range(n_sources)], dtype=object)

    if noise_payload is None:
        if config.chi2_method == "fourier":
            precision_placeholder = np.ones((n_sources, ny, nx, n_bands, 3), dtype=config.dtype_np_real)
            noise_payload = {
                "method": config.noise_psd_method,
                "precision": precision_placeholder,
                "k_indices_y": None,
                "k_indices_x": None,
                "calculator_payload": None,
            }
        else:
            noise_payload = {
                "method": config.noise_psd_method,
                "precision": None,
                "k_indices_y": None,
                "k_indices_x": None,
                "calculator_payload": None,
            }

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
        noise_payload,
    )


def build_dummy_parametric_precision(map_shape, bands, n_sources=2, ell_max=2.5e4, diag_value=1.0):
    """Create a compact precision payload compatible with ParametricPrecisionCalculator."""
    ny, nx = map_shape
    n_bands = len(bands)
    n_stokes = 3
    identity = np.eye(n_bands * n_stokes).reshape(n_bands, n_stokes, n_bands, n_stokes)
    precision_single = identity * diag_value
    precision_grid = np.broadcast_to(precision_single, (ny, nx, n_bands, n_stokes, n_bands, n_stokes))
    precision_stack = np.repeat(precision_grid[None, ...], n_sources, axis=0)

    payload = {
        "precision": precision_stack.astype(np.float64),
        "covariance_model": np.zeros_like(precision_stack.real),
        "covariance_components": {
            "cmb": np.zeros_like(precision_grid),
            "ell_x_model": np.zeros_like(precision_grid),
            "radial_model": np.zeros_like(precision_grid),
        },
        "ell_x_grid": np.zeros((ny, nx), dtype=np.float64),
        "ell_y_grid": np.zeros((ny, nx), dtype=np.float64),
        "ell_radial": np.zeros((ny, nx), dtype=np.float64),
        "ell_max": float(ell_max),
        "metadata": {
            "n_src": precision_stack.shape[0],
            "ny": ny,
            "nx": nx,
            "bands": list(bands),
            "description": "Synthetic precision for testing",
        },
    }

    return payload


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
        gaussfit_yoff = np.array([0.4], dtype=config.dtype_np_real)
        gaussfit_xoff = np.array([-0.6], dtype=config.dtype_np_real)

        maps_clean, _ = loader._prepare_clean_maps(
            gaussfit_amp,
            raw_maps,
            qu_templates,
            source_fields,
            gaussfit_yoff,
            gaussfit_xoff,
        )

        shifted_q = shift_map_bilinear(template[:, :, 0, 0], gaussfit_yoff[0], gaussfit_xoff[0])
        shifted_u = shift_map_bilinear(template[:, :, 0, 1], gaussfit_yoff[0], gaussfit_xoff[0])

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
        noise_payload=None,
    ):
        if config.double_precision:
            jax.config.update("jax_enable_x64", True)
        else:
            jax.config.update("jax_enable_x64", False)

        mock_data = generate_mock_data(config, true_params, noise_payload=noise_payload)
        mock_data_loader.return_value.load_and_prepare.return_value = mock_data

        fitter = PolarizedBeamFitter(config=config)
        best_fit_params = fitter.run_fit()

        print(f"true: {true_params}")
        print(f"best_fit: {best_fit_params['beams'][0]}")
        assertion_func(best_fit_params["beams"], true_params)
        return fitter, best_fit_params

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

    @unittest.skip("B-spline model is currently known to be broken, skipping test...")
    def test_beam_model_bspline(self, *mocks):
        print("\n--- Testing Beam Model: B-spline ---")
        config = get_test_config(beam_model_type="bsplines")
        y, x = np.ogrid[-32:32, -32:32]
        beam_model = create_beam_model(config, y, x, config.bands[0])
        true_coeffs = beam_model.fit_gaussian_coefficients(1.3)
        true_params = {"T_coeffs": true_coeffs, "P_coeffs": true_coeffs}
        self.run_test_and_assert(
            *mocks,
            config,
            true_params,
            lambda fit, true: np.testing.assert_allclose(fit["T_coeffs"], true["T_coeffs"], rtol=0.01),
        )
        print("✓ B-spline model test successful.")

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

    def test_noise_psd_kx_averaged(self, *mocks):
        print("\n--- Testing Noise PSD: kx_averaged ---")
        config = get_test_config(noise_psd_method="kx_averaged")
        true_params = {"beta_pol": 0.75}
        self.run_test_and_assert(
            *mocks,
            config,
            true_params,
            lambda fit, true: self.assertAlmostEqual(fit[0]["beta_pol"], true["beta_pol"], delta=0.01),
        )
        print("✓ kx_averaged noise test successful.")

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

    def test_multiband_parametric_precision(self, *mocks):
        print("\n--- Testing Multi-band with Parametric Precision ---")
        config = get_test_config(
            bands=["90GHz", "150GHz"],
            map_size_pix=12,
            noise_psd_method="parametric_precision",
            n_steps=1500,
        )

        dummy_payload = build_dummy_parametric_precision(
            (config.map_size_pix, config.map_size_pix),
            config.bands,
            n_sources=3,
            ell_max=config.ellmax,
        )

        cached_payload = {
            "method": "parametric_precision",
            "precision": dummy_payload["precision"],
            "k_indices_y": None,
            "k_indices_x": None,
            "calculator_payload": dummy_payload,
        }

        true_params = [{"beta_pol": 0.74}, {"beta_pol": 0.77}]

        def _assert_multiband(fit, expected):
            for band_idx, expected_params in enumerate(expected):
                self.assertAlmostEqual(
                    fit[band_idx]["beta_pol"],
                    expected_params["beta_pol"],
                    delta=0.05,
                )

        self.run_test_and_assert(
            *mocks,
            config,
            true_params,
            _assert_multiband,
            noise_payload=cached_payload,
        )

        print("✓ Multi-band parametric precision test successful.")


class TestUnitNoisePSD(unittest.TestCase):
    """Unit tests for complex functions in the noise_psd module."""

    def test_rebin_psd_with_averaging(self):
        print("\n--- Unit Testing: _rebin_psd_with_averaging ---")
        calc = ClusterfinderPSDCalculator(BeamFittingConfig(), (64, 64))
        psd_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
        expected = np.array([[3.5, 5.5], [11.5, 13.5]])
        rebinned = calc._rebin_psd_with_averaging(psd_array, (2, 2))
        np.testing.assert_allclose(rebinned, expected)
        print("✓ Rebinning test successful.")

    def test_pca_multiband_matches_diagonal_psd(self):
        print("\n--- Unit Testing: PCA multi-band precision vs diagonal PSD estimators ---")
        rng = np.random.default_rng(0)

        config = get_test_config(
            map_size_pix=16,
            bands=["90GHz", "150GHz"],
            noise_psd_method="white_noise",
        )
        config.n_pca_components = 4

        ny = nx = config.map_size_pix
        n_bands = len(config.bands)
        n_stokes = 3
        n_src = 6

        noise_std = np.array([0.3, 0.3, 0.3], dtype=config.dtype_np_real)
        maps_numpy = rng.normal(
            loc=0.0,
            scale=noise_std.reshape(1, 1, 1, 1, n_stokes),
            size=(n_src, ny, nx, n_bands, n_stokes),
        ).astype(config.dtype_np_real)

        ensemble_calc = EnsembleAsdMeanCalculator(config, (ny, nx))
        pca_multi_calc = PcaMultiBandCalculator(config, (ny, nx))

        psd_ensemble = ensemble_calc.calculate_noise_psd(maps_numpy)

        precision_pca_multi = pca_multi_calc.calculate_noise_psd(maps_numpy)
        ny_ax, nx_ax = precision_pca_multi.shape[:2]
        precision_flat = precision_pca_multi.reshape(ny_ax, nx_ax, n_bands * n_stokes, n_bands * n_stokes)
        with np.errstate(divide="ignore", invalid="ignore"):
            psd_from_precision = np.reciprocal(precision_flat)
        psd_from_precision = np.where(np.isfinite(psd_from_precision), psd_from_precision, np.inf)

        diag_indices = np.arange(n_bands * n_stokes)
        psd_diag = psd_from_precision[..., diag_indices, diag_indices].reshape(ny_ax, nx_ax, n_bands, n_stokes)

        ratio = psd_diag / (psd_ensemble + 1e-6)
        self.assertLess(np.median(np.abs(ratio - 1.0)), 0.75)

        diag_vec = np.diagonal(precision_flat, axis1=2, axis2=3)
        identity = np.eye(n_bands * n_stokes, dtype=precision_flat.dtype)
        diag_matrix = diag_vec[..., :, None] * identity  # Broadcast to (ny,nx,N,N)
        precision_offdiag = precision_flat - diag_matrix
        max_offdiag = np.max(np.abs(precision_offdiag))
        max_diag = np.max(np.abs(diag_matrix))
        self.assertLess(max_offdiag, 0.1 * max_diag)
        print("✓ PCA multi-band precision matches diagonal estimators for independent noise.")

    def test_multiband_covariance_axis_order(self):
        print("\n--- Unit Testing: Multi-band covariance axis order ---")
        rng = np.random.default_rng(1)

        config = get_test_config(
            map_size_pix=8,
            bands=["90GHz", "150GHz"],
            noise_psd_method="multiband_covariance",
        )

        ny = nx = config.map_size_pix
        n_bands = len(config.bands)
        n_stokes = 3
        n_src = 3

        maps_numpy = rng.normal(size=(n_src, ny, nx, n_bands, n_stokes)).astype(config.dtype_np_real)

        calc = MultiBandCovarianceCalculator(config, (ny, nx))
        covariance_psd = calc.calculate_noise_psd(maps_numpy)

        self.assertEqual(covariance_psd.shape, (ny, nx, n_bands, n_stokes, n_bands, n_stokes))

        noise_mask = make_apod_mask_center_excised(
            (ny, nx),
            config.apodization_width_pix,
            config.noise_hole_radius_arcmin,
            config.reso_arcmin,
        )
        masked_maps = maps_numpy * noise_mask[None, :, :, None, None]
        masked_maps_fft = np.fft.fft2(masked_maps, axes=(1, 2))
        covariance_sum = np.einsum("nyxbs,nyxct->yxbcst", masked_maps_fft, np.conj(masked_maps_fft))
        effective_area = np.sum(noise_mask**2)
        covariance_expected = covariance_sum / (n_src * effective_area)
        covariance_expected = np.transpose(covariance_expected, (0, 1, 2, 4, 3, 5))

        np.testing.assert_allclose(covariance_psd, covariance_expected)
        print("✓ Multi-band covariance axes interleave band and Stokes indices.")

    def test_parametric_precision_calculator(self):
        print("\n--- Unit Testing: Parametric Precision Calculator ---")
        config = get_test_config(
            map_size_pix=8,
            bands=["90GHz", "150GHz"],
            noise_psd_method="parametric_precision",
        )

        payload = build_dummy_parametric_precision(
            (config.map_size_pix, config.map_size_pix),
            config.bands,
            n_sources=1,
            ell_max=config.ellmax,
        )

        calc = ParametricPrecisionCalculator(
            config,
            (config.map_size_pix, config.map_size_pix),
            precomputed_payload=payload,
        )
        dummy_maps = np.zeros((1, config.map_size_pix, config.map_size_pix, len(config.bands), 3), dtype=config.dtype_np_real)
        precision = calc.calculate_noise_psd(dummy_maps)

        expected_shape = payload["precision"].shape
        self.assertEqual(precision.shape, expected_shape)
        self.assertTrue(np.all(np.isfinite(precision)))
        self.assertTrue(calc.payload is not None)
        self.assertEqual(calc.payload["precision"].shape, precision.shape)
        print("✓ Parametric precision calculator loads precision stack.")


class TestUtilsFunctions(unittest.TestCase):
    """Unit tests for helper utilities."""

    def test_parse_declination_and_nyquist(self):
        decl = parse_declination("J123456-1234.5")
        self.assertAlmostEqual(decl, -12 - 34.5 / 60.0)
        kx = predict_nyquist_kx(decl)
        self.assertTrue(np.isfinite(kx))

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

    def test_compute_2d_asd(self):
        grid = np.zeros((8, 8))
        grid[4, 4] = 1.0
        asd = compute_2d_asd(grid)
        self.assertEqual(asd.shape, grid.shape)
        self.assertGreater(asd[4, 4], 0.0)

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
