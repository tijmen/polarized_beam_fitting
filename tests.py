"""
Tests for the polarized beam fitting package.

To run the tests, use:

```
python -m pytest polarized_beam_fitting/tests.py
```

Author: Tijmen de Haan
Date: 2025-07-03
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import types
import os
import tempfile

from .utils import make_apodization_mask
from .beam_model import BeamModelBspline, BeamModelBetaPol
from .fitter import PolarizedBeamFitter


def get_mock_config(beam_model_type="gaussian", temp_dir=None):
    """
    Creates a mock config object for testing, keeping things like map size fairly small
    powers of 2 to speed up the tests.
    """
    config = types.SimpleNamespace()
    config.map_size_pix = 64
    config.reso_arcmin = 0.1
    config.apodization_width_pix = 5
    config.n_steps = 150000
    config.max_sources = None
    config.pol_focus = 1.0
    config.source_param_names = ["y_offset", "x_offset", "t_amp_factor", "q_amp_factor", "u_amp_factor"]
    config.source_bounds = ((-2.0, 2.0), (-2.0, 2.0), (0.8, 1.2), (-0.01, 0.99), (-10.0, 190.0))
    config.source_inits = (-0.01, -0.01, 1.01, 0.1, 90.1)

    # Beam model parameters
    config.beam_model_type = beam_model_type
    config.bands = ["150GHz"]
    config.band_fwhm_arcmin = {"150GHz": 1.1}
    if beam_model_type == "gaussian":
        config.beam_coeff_bounds = {"T_width_arcmin": (0.8, 1.5), "P_width_arcmin": (0.8, 1.5)}
    elif beam_model_type == "b_spline":
        config.spline_k = 4
        config.spline_rmax_arcmin = 5.0
        config.knot_spacing_arcmin = 0.25
        config.beam_coeff_bounds = {"beam_T_coeffs": (-0.5, 1.5), "beam_P_coeffs": (-0.5, 1.5)}
    elif beam_model_type == "betapol":
        config.beam_coeff_bounds = {"beta_pol": (-0.5, 2.0)}
        config.betapol_data_dir = temp_dir

    return config


def generate_true_params(n_sources=10):
    """
    Make a "params" dictionary for some mock sources.
    """
    np.random.seed(42)  # Ensure tests are reproducible
    true_params = {}
    true_params["fwhm_arcmin"] = 1.1
    true_params["sources"] = {}
    true_params["n_sources"] = n_sources
    true_params["sources"]["y_offset"] = np.random.uniform(-1.0, 0.0, n_sources)
    true_params["sources"]["x_offset"] = np.random.uniform(-1.0, 0.0, n_sources)
    true_params["sources"]["t_amp"] = np.random.uniform(0.95, 1.05, n_sources)
    true_params["sources"]["q_amp"] = np.random.uniform(0.1, 0.3, n_sources)
    true_params["sources"]["u_amp"] = np.random.uniform(0.1, 0.3, n_sources)
    return true_params


def generate_mock_betapol_fitter_data(config, true_params, true_beta_pol, x_grid, y_grid):
    """
    Generates mock data for the fitter using a betapol beam model.
    """
    beam_model = BeamModelBetaPol(config=config, x_grid=x_grid, y_grid=y_grid)

    n_sources = true_params["n_sources"]
    shape = (config.map_size_pix, config.map_size_pix)
    y, x = np.ogrid[-shape[0] // 2 : shape[0] // 2, -shape[1] // 2 : shape[1] // 2]
    apod_mask = make_apodization_mask(shape, config.apodization_width_pix)
    sources_data, source_ids, t_amps, q_amps, u_amps = [], [], [], [], []

    def get_profile_2d(profile_1d_func, dy, dx):
        """Generates a 2D beam map from a 1D profile function."""
        y_shifted_pix = y - dy
        x_shifted_pix = x - dx
        r_arcmin = np.sqrt(y_shifted_pix**2 + x_shifted_pix**2) * config.reso_arcmin
        return profile_1d_func(r_arcmin)

    # Create interpolation functions for the beam profiles
    from scipy.interpolate import interp1d

    # T beam uses BT_r directly
    bt_interp = interp1d(beam_model.r_fine_np, beam_model.BT_r_norm_np, bounds_error=False, fill_value=0.0, kind="linear")

    # P beam is B_main + beta_pol * (B_T - B_main)
    def p_beam_interp(r):
        bmain = interp1d(beam_model.r_fine_np, beam_model.Bmain_r_norm_np, bounds_error=False, fill_value=0.0, kind="linear")(r)
        bt = bt_interp(r)
        return bmain + true_beta_pol * (bt - bmain)

    for i in range(n_sources):
        true_t_amp = true_params["sources"]["t_amp"][i]
        true_q_amp = true_params["sources"]["q_amp"][i]
        true_u_amp = true_params["sources"]["u_amp"][i]
        dy = true_params["sources"]["y_offset"][i]
        dx = true_params["sources"]["x_offset"][i]

        beam_T_map = get_profile_2d(bt_interp, dy, dx)
        beam_P_map = get_profile_2d(p_beam_interp, dy, dx)

        # Create noiseless maps from the true beam profiles
        t_map = true_t_amp * beam_T_map
        q_map = true_q_amp * beam_P_map
        u_map = true_u_amp * beam_P_map

        # Add white noise
        noise_level_T = 0.00005  # Reduced noise level for better recovery
        t_map += np.random.normal(0, noise_level_T, shape)
        q_map += np.random.normal(0, np.sqrt(2) * noise_level_T, shape)
        u_map += np.random.normal(0, np.sqrt(2) * noise_level_T, shape)

        # Apply apodization and FFT
        data_fft = {
            "T": np.fft.fft2(t_map * apod_mask),
            "Q": np.fft.fft2(q_map * apod_mask),
            "U": np.fft.fft2(u_map * apod_mask),
        }

        sources_data.append(data_fft)
        source_ids.append(f"mock_source_{i + 1}")
        t_amps.append(true_t_amp)
        q_amps.append(true_q_amp)
        u_amps.append(true_u_amp)

    return (sources_data, source_ids, t_amps, q_amps, u_amps)


def generate_mock_fitter_data(config, true_params):
    """
    Generates mock data for the fitter.

    This is always a Gaussian beam with FWHM=1.1 arcmin, regardless of whether
    we end up fitting with a Gaussian or B-spline.

    We add a very small amount of white noise.
    """
    n_sources = true_params["n_sources"]
    shape = (config.map_size_pix, config.map_size_pix)
    y, x = np.ogrid[-shape[0] // 2 : shape[0] // 2, -shape[1] // 2 : shape[1] // 2]
    apod_mask = make_apodization_mask(shape, config.apodization_width_pix)
    true_fwhm = true_params["fwhm_arcmin"]
    sigma_pix = true_fwhm / (config.reso_arcmin * 2.355)
    sources_data, source_ids, t_amps, q_amps, u_amps = [], [], [], [], []

    for i in range(n_sources):
        true_t_amp = true_params["sources"]["t_amp"][i]
        true_q_amp = true_params["sources"]["q_amp"][i]
        true_u_amp = true_params["sources"]["u_amp"][i]
        dy = true_params["sources"]["y_offset"][i]
        dx = true_params["sources"]["x_offset"][i]

        beam_T_map = beam_P_map = np.exp(-((x - dx) ** 2 + (y - dy) ** 2) / (2 * sigma_pix**2))

        # Create noiseless maps from the true beam profiles
        t_map = true_t_amp * beam_T_map
        q_map = true_q_amp * beam_P_map
        u_map = true_u_amp * beam_P_map

        # Add white noise
        noise_level_T = 0.0001
        t_map += np.random.normal(0, noise_level_T, shape)
        q_map += np.random.normal(0, np.sqrt(2) * noise_level_T, shape)
        u_map += np.random.normal(0, np.sqrt(2) * noise_level_T, shape)

        # Apply apodization and FFT
        data_fft = {
            "T": np.fft.fft2(t_map * apod_mask),
            "Q": np.fft.fft2(q_map * apod_mask),
            "U": np.fft.fft2(u_map * apod_mask),
        }

        sources_data.append(data_fft)
        source_ids.append(f"mock_source_{i + 1}")
        t_amps.append(true_t_amp)
        q_amps.append(true_q_amp)
        u_amps.append(true_u_amp)

    return (sources_data, source_ids, t_amps, q_amps, u_amps)


class TestBeamModelBspline(unittest.TestCase):
    """
    Tests the BeamModelBspline class, which is a critical and complex component.
    Ensures the B-spline basis functions are generated correctly and can
    represent a Gaussian pretty accurately.
    """

    def setUp(self):
        self.config = get_mock_config(beam_model_type="b_spline")
        shape = (self.config.map_size_pix, self.config.map_size_pix)
        y, x = np.ogrid[-shape[0] // 2 : shape[0] // 2, -shape[1] // 2 : shape[1] // 2]
        self.x_grid, self.y_grid = np.meshgrid(x, y)
        self.beam_model = BeamModelBspline(
            config=self.config,
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            spline_k=self.config.spline_k,
            spline_rmax_arcmin=self.config.spline_rmax_arcmin,
            knot_spacing_arcmin=self.config.knot_spacing_arcmin,
            fwhm_arcmin=self.config.band_fwhm_arcmin["150GHz"],
        )

    def test_fit_gaussian_coefficients(self):
        """
        Verify that the B-spline basis can accurately model a simple Gaussian.
        This is a key validation of the basis function generation.
        """
        fwhm_to_fit = 1.4152  # just some random-ish number
        coeffs = self.beam_model.fit_gaussian_coefficients(fwhm_to_fit)

        # Reconstruct the profile from the fitted coefficients
        reconstructed_profile = self.beam_model.evaluate_beam_profile(coeffs, self.beam_model.r_fine_jax)

        # Create the target Gaussian profile
        sigma = fwhm_to_fit / 2.355
        target_gaussian = np.exp(-0.5 * (self.beam_model.r_fine_jax / sigma) ** 2)

        # The reconstructed profile should be very close to the target
        # Use a Kolmogorov-Smirnov-like test: the maximum deviation should be small.
        max_deviation = np.max(np.abs(reconstructed_profile - target_gaussian))
        self.assertLess(max_deviation, 0.01, "B-spline basis should approximate a Gaussian with <1% max error")

        # The profile should be normalized to 1 at r=0
        self.assertAlmostEqual(reconstructed_profile[0], 1.0, places=5)


@patch("polarized_beam_fitting.fitter.create_noise_psd_calculator")
@patch("polarized_beam_fitting.fitter.PolarizedBeamFitter._load_and_prepare_data")
class TestFitterRecovery(unittest.TestCase):
    """
    High-level integration tests to verify parameter recovery.
    These are the most important tests, ensuring the entire fitting pipeline
    works correctly for either the Gaussian or B-spline model.
    """

    def run_recovery_test(self, mock_load_data, mock_create_noise, config, true_params):
        """A generic test runner to avoid code duplication."""
        # Arrange: Create mock data from true parameters and mock the dependencies
        shape = (config.map_size_pix, config.map_size_pix)
        y, x = np.ogrid[-shape[0] // 2 : shape[0] // 2, -shape[1] // 2 : shape[1] // 2]
        x_grid, y_grid = np.meshgrid(x, y)
        mock_load_data.return_value = generate_mock_fitter_data(config, true_params)

        mock_noise_calc = MagicMock()
        mock_noise_calc.is_individual_psds.return_value = False
        shape = (config.map_size_pix, config.map_size_pix)
        # Use a simple, low-noise PSD to make the fit deterministic
        mock_noise_calc.calculate_noise_psd.return_value = {
            "TT": np.ones(shape),
            "QQ": np.ones(shape) * 2,
            "UU": np.ones(shape) * 2,  # the polarization maps always have roughly twice the noise power of the temperature map
        }
        mock_create_noise.return_value = mock_noise_calc

        # Act: Initialize the fitter and run the optimization
        fitter = PolarizedBeamFitter(config=config)

        # A robust fitter should converge from default initial parameters
        best_fit_params = fitter.run_fit()
        return fitter, best_fit_params

    def test_gaussian_parameter_recovery(self, mock_load_data, mock_create_noise):
        """
        Verify the fitter can recover the FWHM and source properties for a Gaussian model.
        """
        print("\n--- Running Gaussian Parameter Recovery Test ---")
        config = get_mock_config(beam_model_type="gaussian")
        true_params = generate_true_params(n_sources=10)

        _, best_fit_params = self.run_recovery_test(mock_load_data, mock_create_noise, config, true_params)

        # Assert: Check if recovered parameters match the true ones
        # 1. Beam FWHM
        self.assertAlmostEqual(best_fit_params["beam"]["T_width_arcmin"], true_params["fwhm_arcmin"], delta=0.01)
        self.assertAlmostEqual(best_fit_params["beam"]["P_width_arcmin"], true_params["fwhm_arcmin"], delta=0.01)

        # 2. TQU amplitudes
        t_amp_factor_err = np.mean(np.abs(best_fit_params["sources"]["t_amp_factor"] - true_params["sources"]["t_amp_factor"]))
        q_amp_factor_err = np.mean(np.abs(best_fit_params["sources"]["q_amp_factor"] - true_params["sources"]["q_amp_factor"]))
        u_amp_factor_err = np.mean(np.abs(best_fit_params["sources"]["u_amp_factor"] - true_params["sources"]["u_amp_factor"]))
        self.assertLess(t_amp_factor_err, 0.01, "T amplitude factor should be recovered accurately")
        self.assertLess(q_amp_factor_err, 0.01, "Q amplitude factor should be recovered accurately")
        self.assertLess(u_amp_factor_err, 0.01, "U amplitude factor should be recovered accurately")

        print("✓ Gaussian recovery successful.")

    def test_bspline_profile_recovery(self, mock_load_data, mock_create_noise):
        """
        Verify the fitter can recover a B-spline beam profile that
        accurately matches the input Gaussian beam profile.
        """
        print("\n--- Running B-spline Profile Recovery Test ---")
        config = get_mock_config(beam_model_type="b_spline")

        true_params = generate_true_params(n_sources=10)

        fitter, best_fit_params = self.run_recovery_test(mock_load_data, mock_create_noise, config, true_params)

        true_profile_T = fitter.beam_model.evaluate_beam_profile(fitter.beam_model.fit_gaussian_coefficients(true_params["fwhm_arcmin"]), fitter.beam_model.r_fine_jax)
        fit_profile_T = fitter.beam_model.evaluate_beam_profile(best_fit_params["beam"]["beam_T_coeffs"], fitter.beam_model.r_fine_jax)
        max_dev_T = np.max(np.abs(true_profile_T - fit_profile_T))
        self.assertLess(max_dev_T, 0.01, "T beam profile max deviation should be < 1%")

        true_profile_P = fitter.beam_model.evaluate_beam_profile(fitter.beam_model.fit_gaussian_coefficients(true_params["fwhm_arcmin"]), fitter.beam_model.r_fine_jax)
        fit_profile_P = fitter.beam_model.evaluate_beam_profile(best_fit_params["beam"]["beam_P_coeffs"], fitter.beam_model.r_fine_jax)
        max_dev_P = np.max(np.abs(true_profile_P - fit_profile_P))
        self.assertLess(max_dev_P, 0.01, "P beam profile max deviation should be < 1%")
        print("✓ B-spline recovery successful.")

    def test_betapol_parameter_recovery(self, mock_load_data, mock_create_noise):
        """
        Verify the fitter can recover the beta_pol parameter and source properties.
        """
        print("\n--- Running Betapol Parameter Recovery Test ---")

        # 1. Create mock betapol data
        band = "150GHz"
        r_arcmin = np.linspace(0, 10, 1000)

        # B_main is a Gaussian with FWHM = 1.0'
        sigma_main = 1.0 / 2.355
        Bmain_r = np.exp(-0.5 * (r_arcmin / sigma_main) ** 2)

        # B_T is a Gaussian with FWHM = 1.2'
        sigma_T = 1.2 / 2.355
        BT_r = np.exp(-0.5 * (r_arcmin / sigma_T) ** 2)

        # Create mock beam data dictionary
        mock_beam_data = {"r_fine_arcmin": r_arcmin, "Bmain_r_norm_150": Bmain_r, "BT_r_norm_150": BT_r}

        # 2. Patch np.load to return our mock data when betapol.npz is loaded
        original_load = np.load

        def mock_load(filename, *args, **kwargs):
            if "betapol_TdH.npz" in str(filename):
                # Return a mock object that behaves like numpy's loaded data
                class MockNumpyData:
                    def __init__(self, data):
                        self._data = data

                    def keys(self):
                        return self._data.keys()

                    def __getitem__(self, key):
                        return self._data[key]

                    def __contains__(self, key):
                        return key in self._data

                    def __getattr__(self, key):
                        if key in self._data:
                            return self._data[key]
                        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

                return MockNumpyData(mock_beam_data)
            else:
                return original_load(filename, *args, **kwargs)

        with patch("numpy.load", side_effect=mock_load):
            # 3. Setup config and true parameters
            config = get_mock_config(beam_model_type="betapol")
            true_params = generate_true_params(n_sources=10)
            true_beta_pol = 0.6

            shape = (config.map_size_pix, config.map_size_pix)
            y, x = np.ogrid[-shape[0] // 2 : shape[0] // 2, -shape[1] // 2 : shape[1] // 2]
            x_grid, y_grid = np.meshgrid(x, y)

            # 4. Generate mock data using the betapol model
            mock_data = generate_mock_betapol_fitter_data(config, true_params, true_beta_pol, x_grid, y_grid)
            mock_load_data.return_value = mock_data

            # 5. Run the recovery test
            fitter, best_fit_params = self.run_recovery_test(mock_load_data, mock_create_noise, config, true_params)

            # 6. Assert recovery (use slightly looser tolerance for betapol model)
            self.assertAlmostEqual(best_fit_params["beam"]["beta_pol"], true_beta_pol, delta=0.08)

            # also check source params
            t_amp_factor_err = np.mean(np.abs(best_fit_params["sources"]["t_amp_factor"] - true_params["sources"]["t_amp_factor"]))
            q_amp_factor_err = np.mean(np.abs(best_fit_params["sources"]["q_amp_factor"] - true_params["sources"]["q_amp_factor"]))
            u_amp_factor_err = np.mean(np.abs(best_fit_params["sources"]["u_amp_factor"] - true_params["sources"]["u_amp_factor"]))
            self.assertLess(t_amp_factor_err, 0.01, "T amplitude factor should be recovered accurately")
            self.assertLess(q_amp_factor_err, 0.01, "Q amplitude factor should be recovered accurately")
            self.assertLess(u_amp_factor_err, 0.01, "U amplitude factor should be recovered accurately")

            print("✓ Betapol recovery successful.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
