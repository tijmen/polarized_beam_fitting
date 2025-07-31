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
import tempfile

from .utils import make_apodization_mask
from .beam_model import BeamModelBspline, BeamModelBetaPol
from .fitter import PolarizedBeamFitter
from .config import BeamFittingConfig


def get_test_config(beam_model_type="gaussian"):
    """
    Creates a test config using the actual BeamFittingConfig class,
    with overrides for fast testing.
    """
    config = BeamFittingConfig()
    
    # Override for fast testing
    config.map_size_pix = 64
    config.reso_arcmin = 0.1
    config.apodization_width_pix = 5
    config.n_steps = 5000
    config.noise_psd_method = "white_noise"
    config.bands = ["150GHz"]
    
    # Use temporary directories
    config.cache_dir = tempfile.mkdtemp()
    config.coadd_filenames = ["mock_file.g3"]
    
    # Set beam model type
    config.beam_model_type = beam_model_type
    if beam_model_type == "b_spline":
        config.spline_rmax_arcmin = 5.0
        config.knot_spacing_arcmin = 0.25
    
    return config


def generate_mock_data(config, n_sources=10):
    """
    Generate mock data in the expected array format.
    
    Returns the 9-tuple expected by PolarizedBeamFitter:
    (gaussfit_yoff, gaussfit_xoff, gaussfit_initial_amp, raw_maps, 
     normalized_qu_templates, maps_numpy, maps_fft_numpy, source_ids, n_src)
    """
    np.random.seed(42)  # Ensure tests are reproducible
    
    # Generate mock source parameters
    yoff = np.random.uniform(-1.0, 0.0, n_sources)
    xoff = np.random.uniform(-1.0, 0.0, n_sources)
    amp_factors = np.random.uniform([0.95, 0.1, 0.1], [1.05, 0.3, 0.3], (n_sources, 3))  # [T, Q, U]
    
    # Create mock beam maps with Gaussian profile
    shape = (config.map_size_pix, config.map_size_pix)
    y, x = np.ogrid[-shape[0]//2:shape[0]//2, -shape[1]//2:shape[1]//2]
    apod_mask = make_apodization_mask(shape, config.apodization_width_pix)
    
    fwhm_arcmin = config.band_fwhm_arcmin[config.bands[0]]
    sigma_pix = fwhm_arcmin / (config.reso_arcmin * 2.355)
    
    n_bands = len(config.bands)
    n_stokes = 3
    
    # Initialize arrays with shape (n_source, ny, nx, n_bands, n_stokes)
    maps_numpy = np.zeros((n_sources, shape[0], shape[1], n_bands, n_stokes))
    maps_fft_numpy = np.zeros((n_sources, shape[0], shape[1], n_bands, n_stokes), dtype=complex)
    
    for i in range(n_sources):
        # Create Gaussian beam shifted by source offset
        beam_map = np.exp(-((x - xoff[i]) ** 2 + (y - yoff[i]) ** 2) / (2 * sigma_pix**2))
        
        for band_idx in range(n_bands):
            for stokes_idx in range(n_stokes):
                # Apply amplitude factor and add noise
                signal_map = amp_factors[i, stokes_idx] * beam_map
                noise_level = 0.0001 * (1.0 if stokes_idx == 0 else np.sqrt(2))
                signal_map += np.random.normal(0, noise_level, shape)
                
                # Apply apodization
                apodized_map = signal_map * apod_mask
                maps_numpy[i, :, :, band_idx, stokes_idx] = apodized_map
                maps_fft_numpy[i, :, :, band_idx, stokes_idx] = np.fft.fft2(apodized_map)
    
    # Prepare other required arrays
    gaussfit_yoff = np.tile(yoff[:, np.newaxis], (1, n_bands))  # (n_src, n_bands)
    gaussfit_xoff = np.tile(xoff[:, np.newaxis], (1, n_bands))  # (n_src, n_bands)
    gaussfit_initial_amp = np.tile(amp_factors[:, np.newaxis, :], (1, n_bands, 1))  # (n_src, n_bands, 3)
    
    # Mock data - these aren't used in the optimization but need to exist
    raw_maps = np.zeros_like(maps_numpy)
    normalized_qu_templates = np.zeros((shape[0], shape[1], n_bands, 2))  # Only Q, U
    source_ids = [f"mock_source_{i+1}" for i in range(n_sources)]
    
    return (gaussfit_yoff, gaussfit_xoff, gaussfit_initial_amp, raw_maps, 
            normalized_qu_templates, maps_numpy, maps_fft_numpy, source_ids, n_sources)


def create_mock_noise_psd(config):
    """Create a simple white noise PSD for testing."""
    shape = (config.map_size_pix, config.map_size_pix)
    n_bands = len(config.bands)
    noise_psd_array = np.zeros((shape[0], shape[1], n_bands, 3), dtype=np.float32)
    noise_psd_array[:, :, 0, 0] = 1.0  # T
    noise_psd_array[:, :, 0, 1] = 2.0  # Q (2x noise)
    noise_psd_array[:, :, 0, 2] = 2.0  # U (2x noise)
    return noise_psd_array


class TestBeamModelBspline(unittest.TestCase):
    """Test B-spline beam model functionality."""

    def setUp(self):
        self.config = get_test_config(beam_model_type="b_spline")
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
        )

    def test_fit_gaussian_coefficients(self):
        """Verify B-spline basis can accurately model a Gaussian."""
        fwhm_to_fit = 1.4152
        coeffs = self.beam_model.fit_gaussian_coefficients(fwhm_to_fit)
        reconstructed_profile = self.beam_model.evaluate_beam_profile(coeffs, self.beam_model.r_fine_jax)
        
        sigma = fwhm_to_fit / 2.355
        target_gaussian = np.exp(-0.5 * (self.beam_model.r_fine_jax / sigma) ** 2)
        
        max_deviation = np.max(np.abs(reconstructed_profile - target_gaussian))
        self.assertLess(max_deviation, 0.01, "B-spline should approximate Gaussian with <1% error")
        self.assertAlmostEqual(reconstructed_profile[0], 1.0, places=5)


@patch("polarized_beam_fitting.fitter.create_noise_psd_calculator")
@patch("polarized_beam_fitting.fitter.PolarizedBeamFitter._load_and_prepare_data")
class TestFitterRecovery(unittest.TestCase):
    """Test parameter recovery for different beam models."""

    def run_recovery_test(self, mock_load_data, mock_create_noise, beam_model_type, n_sources=5):
        """Generic test runner."""
        config = get_test_config(beam_model_type)
        mock_data = generate_mock_data(config, n_sources)
        mock_load_data.return_value = mock_data
        
        mock_noise_calc = MagicMock()
        mock_noise_calc.is_individual_psds.return_value = False
        mock_noise_calc.calculate_noise_psd.return_value = create_mock_noise_psd(config)
        mock_create_noise.return_value = mock_noise_calc
        
        fitter = PolarizedBeamFitter(config=config)
        best_fit_params = fitter.run_fit()
        return fitter, best_fit_params, mock_data

    def test_gaussian_parameter_recovery(self, mock_load_data, mock_create_noise):
        """Test Gaussian beam parameter recovery."""
        print("\n--- Testing Gaussian Parameter Recovery ---")
        
        fitter, best_fit_params, mock_data = self.run_recovery_test(
            mock_load_data, mock_create_noise, "gaussian"
        )
        
        # Check beam parameters
        true_fwhm = fitter.config.band_fwhm_arcmin[fitter.config.bands[0]]
        self.assertAlmostEqual(best_fit_params["beam"]["T_width_arcmin"], true_fwhm, delta=0.05)
        self.assertAlmostEqual(best_fit_params["beam"]["P_width_arcmin"], true_fwhm, delta=0.05)
        
        print("✓ Gaussian recovery successful.")

    def test_bspline_profile_recovery(self, mock_load_data, mock_create_noise):
        """Test B-spline beam profile recovery."""
        print("\n--- Testing B-spline Profile Recovery ---")
        
        fitter, best_fit_params, mock_data = self.run_recovery_test(
            mock_load_data, mock_create_noise, "b_spline"
        )
        
        # Check that profiles are reasonable (not exact due to noise and optimization)
        true_fwhm = fitter.config.band_fwhm_arcmin[fitter.config.bands[0]]
        true_profile = fitter.beam_model.evaluate_beam_profile(
            fitter.beam_model.fit_gaussian_coefficients(true_fwhm), 
            fitter.beam_model.r_fine_jax
        )
        fit_profile_T = fitter.beam_model.evaluate_beam_profile(
            best_fit_params["beam"]["beam_T_coeffs"], 
            fitter.beam_model.r_fine_jax
        )
        
        max_dev = np.max(np.abs(true_profile - fit_profile_T))
        self.assertLess(max_dev, 0.05, "B-spline should approximate target profile")
        
        print("✓ B-spline recovery successful.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
