"""
Tests for the polarized beam fitting package.

This test suite includes both end-to-end "round-trip" tests and focused
unit tests. The end-to-end tests verify that the fitter can recover known
input beam parameters under various configurations, exploring one feature
at a time away from a default setup. Unit tests cover complex, isolated
functions.

To run all tests, use:
`python -m pytest polarized_beam_fitting/tests.py`

Author: Tijmen de Haan
Date: 2025-08-02
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import jax
import numpy as np
from astropy.io import fits

from polarized_beam_fitting.beam_model import create_beam_model
from polarized_beam_fitting.config import BeamFittingConfig
from polarized_beam_fitting.fitter import PolarizedBeamFitter
from polarized_beam_fitting.noise_psd import ClusterfinderPSDCalculator
from polarized_beam_fitting.utils import make_apodization_mask


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
        np.savez(
            dummy_betapol_file,
            r_fine_arcmin=r_fine,
            BT_r_norm_150=bt,
            Bmain_r_norm_150=bmain,
        )
        config.betapol_data_path = dummy_betapol_file

    if config.noise_psd_method == "clusterfinder_psd":
        psd_file = tempfile.NamedTemporaryFile(delete=False, suffix=".fits").name
        psd_data = np.ones((256, 256), dtype=config.dtype_np_real)
        fits.writeto(psd_file, psd_data, overwrite=True)
        config.noise_psd_path = psd_file

    return config


def generate_mock_data(config, true_beam_params, n_sources=3):
    """
    Generate mock data based on a specific beam model and true parameters.
    """
    np.random.seed(42)
    shape = (config.map_size_pix, config.map_size_pix)
    ny, nx = shape
    y_grid, x_grid = np.ogrid[-ny // 2 : ny // 2, -nx // 2 : nx // 2]

    beam_model = create_beam_model(config, y_grid, x_grid, config.bands[0])

    true_yoffs = np.random.uniform(-1.5, 1.5, n_sources)
    true_xoffs = np.random.uniform(-1.5, 1.5, n_sources)
    true_amps = np.random.uniform([0.9, -0.05, -0.05], [1.1, 0.05, 0.05], (n_sources, 3))

    n_bands = len(config.bands)
    maps_numpy = np.zeros((n_sources, ny, nx, n_bands, 3), dtype=config.dtype_np_real)
    weights_numpy = np.zeros((n_sources, ny, nx, n_bands, 3, 3), dtype=config.dtype_np_real)

    for i in range(n_sources):
        true_beam_T, true_beam_P = beam_model.evaluate_beam_maps(true_beam_params, true_yoffs[i], true_xoffs[i])
        signal_maps = np.stack(
            [
                true_amps[i, 0] * true_beam_T,
                true_amps[i, 1] * true_beam_P,
                true_amps[i, 2] * true_beam_P,
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

        maps_numpy[i, :, :, 0, :] = signal_maps + noise_maps
        weights_numpy[i, :, :, 0, 0, 0] = 1.0 / (noise_level**2)
        weights_numpy[i, :, :, 0, 1, 1] = 1.0 / ((noise_level * np.sqrt(2)) ** 2)
        weights_numpy[i, :, :, 0, 2, 2] = 1.0 / ((noise_level * np.sqrt(2)) ** 2)

    apod_mask = make_apodization_mask(shape, config.apodization_width_pix)
    maps_apodized = maps_numpy * apod_mask[np.newaxis, :, :, np.newaxis, np.newaxis]
    maps_fft_numpy = np.fft.fft2(maps_apodized, axes=(1, 2))

    gaussfit_yoff = true_yoffs
    gaussfit_xoff = true_xoffs
    gaussfit_initial_amp = true_amps[:, np.newaxis, :]

    raw_maps = np.zeros_like(maps_numpy)
    qu_templates = np.zeros((ny, nx, n_bands, 2))
    source_ids = np.array([f"mock_source_{i}" for i in range(n_sources)])

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
        n_sources,
    )


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
