"""End-to-end and unit tests for the polarized beam fitting package."""

import os
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")

from polarized_beam_fitting.beam_model import create_beam_model
from polarized_beam_fitting.config import BeamFittingConfig
from polarized_beam_fitting.data_loader import DataLoader
from polarized_beam_fitting.fitter import PolarizedBeamFitter
from polarized_beam_fitting.plotting import BeamPlotter, create_diagnostic_plots
from polarized_beam_fitting.utils import (
    calculate_tod_nyquist_radial_mask_smooth,
    compute_rectangular_ell_cut_indices,
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
    config.reso_arcmin = 0.25
    config.apodization_width_pix = 8
    config.bands = ["150GHz"]
    config.min_t_amplitude = 0
    config.cache_dir = tempfile.mkdtemp()
    config.coadd_filenames = {"test_field": ["mock_file.g3"]}
    config.n_diagnostic_plots = 0
    config.ellmax = 20000
    config.bfgs_kwargs = {"atol": 1e-24, "rtol": 1e-24, "verbose": frozenset({})}

    # --- Apply user-specified overrides ---
    for key, value in kwargs.items():
        setattr(config, key, value)

    # --- Handle special cases based on config ---
    if config.beam_model_type == "beta_pol":
        dummy_betapol_file = os.path.join(config.cache_dir, "betapol_tests.npz")
        r_fine = np.linspace(0, 10, 100)
        sigma_main, sigma_bt = 2.0 / 2.355, 3.2 / 2.355
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


def generate_mock_data(config, true_beam_params, n_sources=10):
    """
    Generate mock data based on a specific beam model and true parameters.
    Draws separate true and initial parameters for sources to test recovery.
    """
    np.random.seed(42)
    shape = (config.map_size_pix, config.map_size_pix)
    ny_full, nx_full = shape
    y_grid, x_grid = np.ogrid[-ny_full // 2 : ny_full // 2, -nx_full // 2 : nx_full // 2]

    band_params = _normalize_true_params(config, true_beam_params)
    n_bands = len(config.bands)
    beam_models = [create_beam_model(config, y_grid, x_grid, band) for band in config.bands]

    # Draw TRUE source parameters - all with shape appropriate for multi-band
    true_yoffs = np.random.uniform(-0.6, -0.4, (n_sources, n_bands))
    true_xoffs = np.random.uniform(-0.6, -0.4, (n_sources, n_bands))
    amp_low = np.array([0.9, -0.05, -0.05])
    amp_high = np.array([1.1, 0.05, 0.05])
    true_amps = amp_low + (amp_high - amp_low) * np.random.uniform(size=(n_sources, n_bands, 3))

    # Draw INITIAL source parameters (different from true)
    init_yoffs = np.random.uniform(-0.6, -0.4, (n_sources, n_bands))
    init_xoffs = np.random.uniform(-0.6, -0.4, (n_sources, n_bands))
    init_amps = amp_low + (amp_high - amp_low) * np.random.uniform(size=(n_sources, n_bands, 3))

    maps_numpy = np.zeros((n_sources, ny_full, nx_full, n_bands, 3), dtype=config.dtype_np_real)
    weights_numpy = np.zeros((n_sources, ny_full, nx_full, n_bands, 3, 3), dtype=config.dtype_np_real)
    noise_sigma_T = 1e-4
    noise_sigma_P = noise_sigma_T * np.sqrt(2.0)

    for i in range(n_sources):
        for band_idx, beam_model in enumerate(beam_models):
            true_beam_T, true_beam_P = beam_model.evaluate_beam_maps(
                band_params[band_idx], true_yoffs[i, band_idx], true_xoffs[i, band_idx]
            )
            signal_maps = np.stack(
                [
                    true_amps[i, band_idx, 0] * true_beam_T,
                    true_amps[i, band_idx, 1] * true_beam_P,
                    true_amps[i, band_idx, 2] * true_beam_P,
                ],
                axis=-1,
            )

            noise_maps = np.stack(
                [
                    np.random.normal(0, noise_sigma_T, shape),
                    np.random.normal(0, noise_sigma_P, shape),
                    np.random.normal(0, noise_sigma_P, shape),
                ],
                axis=-1,
            )

            maps_numpy[i, :, :, band_idx, :] = signal_maps + noise_maps
            weights_numpy[i, :, :, band_idx, 0, 0] = 1.0 / (noise_sigma_T**2)
            weights_numpy[i, :, :, band_idx, 1, 1] = 1.0 / (noise_sigma_P**2)
            weights_numpy[i, :, :, band_idx, 2, 2] = 1.0 / (noise_sigma_P**2)

    apod_mask = make_apodization_mask(shape, config.apodization_width_pix)
    maps_apodized = maps_numpy * apod_mask[np.newaxis, :, :, np.newaxis, np.newaxis]
    maps_fft_numpy = np.fft.fft2(maps_apodized, axes=(1, 2))

    # Return INITIAL parameters (different from true) for the fitter to start from
    gaussfit_yoff = init_yoffs
    gaussfit_xoff = init_xoffs
    gaussfit_initial_amp = init_amps

    raw_maps = np.zeros_like(maps_numpy)
    qu_templates = np.zeros((ny_full, nx_full, n_bands, 2))
    source_ids = np.array([f"mock_source_{i}" for i in range(n_sources)])
    source_fields = np.array(["mock_field" for _ in range(n_sources)], dtype=object)

    if config.chi2_method == "fourier":
        # Apply the same ell truncation as the real data loader
        idx_y, idx_x = compute_rectangular_ell_cut_indices((ny_full, nx_full), config.reso_arcmin, config.ellmax)
        precision_template = np.zeros((n_bands, 3, n_bands, 3), dtype=config.dtype_np_real)
        for band_idx in range(n_bands):
            precision_template[band_idx, 0, band_idx, 0] = 1.0
            precision_template[band_idx, 1, band_idx, 1] = 1.0
            precision_template[band_idx, 2, band_idx, 2] = 1.0

        if idx_y is not None and idx_x is not None:
            ny, nx = len(idx_y), len(idx_x)
            # Truncate ONLY the FFT maps - real-space maps stay full size
            maps_fft_numpy = maps_fft_numpy[:, idx_y, :, :, :][:, :, idx_x, :, :]
            precision_placeholder = np.broadcast_to(precision_template, (n_sources, ny, nx, n_bands, 3, n_bands, 3)).copy()
        else:
            precision_placeholder = np.broadcast_to(precision_template, (n_sources, ny_full, nx_full, n_bands, 3, n_bands, 3)).copy()
        debug_placeholder = None
    else:
        precision_placeholder = None
        debug_placeholder = None

    # Pack true parameters for testing recovery
    true_source_params = {
        "yoff": true_yoffs,
        "xoff": true_xoffs,
        "flux": true_amps,
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
        precision_placeholder,
        debug_placeholder,
        true_source_params,
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
        true_beam_params,
        assertion_func,
    ):
        if config.double_precision:
            jax.config.update("jax_enable_x64", True)
        else:
            jax.config.update("jax_enable_x64", False)

        mock_data = generate_mock_data(config, true_beam_params)
        true_source_params = mock_data[-1]  # Last element is true_source_params
        # Pass only the first 13 elements to fitter (exclude true_source_params)
        mock_data_loader.return_value.load_and_prepare.return_value = mock_data[:-1]

        fitter = PolarizedBeamFitter(config=config)
        best_fit_params = fitter.run_fit()

        print(f"\ntrue beam: {true_beam_params}")
        print(f"best_fit beam: {best_fit_params['beams'][0]}")

        # Test beam parameter recovery
        assertion_func(best_fit_params["beams"], true_beam_params)

        # Test source parameter recovery
        yoff_error = np.sqrt(np.mean((np.array(best_fit_params["sources"]["yoff"]) - true_source_params["yoff"]) ** 2))
        xoff_error = np.sqrt(np.mean((np.array(best_fit_params["sources"]["xoff"]) - true_source_params["xoff"]) ** 2))
        flux_error = np.sqrt(np.mean((np.array(best_fit_params["sources"]["flux"]) - true_source_params["flux"]) ** 2))

        print("Source parameter recovery (RMS errors):")
        print(f"  yoff: {yoff_error:.6f} pixels (tolerance: 0.01)")
        print(f"  xoff: {xoff_error:.6f} pixels (tolerance: 0.01)")
        print(f"  flux: {flux_error:.6f} (tolerance: 0.01)")

        self.assertLess(yoff_error, 0.01, f"yoff recovery failed: RMS error = {yoff_error:.6f}")
        self.assertLess(xoff_error, 0.01, f"xoff recovery failed: RMS error = {xoff_error:.6f}")
        self.assertLess(flux_error, 0.01, f"flux recovery failed: RMS error = {flux_error:.6f}")

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
        config = get_test_config(chi2_method="real_space")
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

    def test_solver_bfgs(self, *mocks):
        print("\n--- Testing Solver: Optimistix BFGS ---")
        config = get_test_config(solver="optimistix_bfgs")
        config.bfgs_kwargs = {"atol": 1e-18, "rtol": 1e-18, "verbose": frozenset({})}
        true_params = {"beta_pol": 0.75}
        self.run_test_and_assert(
            *mocks,
            config,
            true_params,
            lambda fit, true: self.assertAlmostEqual(fit[0]["beta_pol"], true["beta_pol"], delta=0.02),
        )
        print("✓ BFGS solver test successful.")


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
        config = get_test_config(map_size_pix=12)
        config.n_diagnostic_plots = 1
        config.skip_sources = []

        with tempfile.TemporaryDirectory() as temp_root:
            config.output_dir = os.path.join(temp_root, "default_plots")

            mock_data_loader.return_value.load_and_prepare.return_value = generate_mock_data(
                config,
                {"beta_pol": 0.72},
                n_sources=2,
            )[:-1]

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
