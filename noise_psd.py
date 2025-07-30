"""
Noise Power Spectral Density (PSD) implementations for polarized beam fitting.

This module provides different approaches to estimating noise PSDs used in the
maximum-likelihood fitting of polarized beams.

There are several approaches for calculating the noise PSD. One of the major
differences is whether the noise PSD is fully diagonal (each ky,kx,band,stokes
is separate) or only diagonal in Fourier space (ky,ky independent, but band-band
and stokes-stokes off-diagonals).
We will use config.noise_psd_method to decide.
Currently, [clusterfinder_psd, kx_averaged_individual, white_noise_scaled, ensemble_asd_mean]
are fully diagonal, and [multiband_covariance] is only diagonal in Fourier space.
"""

import numpy as np
from abc import ABC, abstractmethod
from astropy.io import fits
from .utils import make_apod_mask_center_excised


class NoisePSDCalculator(ABC):
    """
    Abstract base class for noise PSD calculators.

    Different implementations can inherit from this class to provide
    various methods of estimating noise power spectral densities.
    """

    def __init__(self, config, map_shape):
        """
        Initialize the noise PSD calculator.

        Parameters:
        -----------
        config : BeamFittingConfig
            Configuration object
        map_shape : tuple
            Shape of the maps (ny, nx)
        """
        self.config = config
        self.bands = config.bands
        self.n_bands = len(self.bands)
        self.map_shape = map_shape

    @abstractmethod
    def calculate_noise_psd(self, prepared_data_py):
        """
        Calculate noise PSD(s) for the given data.

        Parameters:
        -----------
        prepared_data_py : list
            List of dictionaries containing FFT data for each source

        Returns:
        --------
        np.ndarray or list
            For single-band: (ky,kx,band,stokes) array for global PSD,
            or list of such arrays for individual PSDs per source.
            For multi-band: (ky,kx,band,band,stokes,stokes) array for covariance.
        """
        pass

    @abstractmethod
    def is_individual_psds(self):
        """
        Return whether this calculator provides individual PSDs per source.

        Returns:
        --------
        bool
            True if individual PSDs per source, False if global PSD
        """
        pass


class ClusterfinderPSDCalculator(NoisePSDCalculator):
    """
    Load pre-computed instrument noise PSD from clusterfinder analysis.

    This implementation loads a noise PSD from a FITS file that was pre-computed
    from clusterfinder instrument characterization data, then resamples it to match
    the analysis map resolution using mean-pooling.
    """

    def calculate_noise_psd(self, prepared_data_py):
        """
        Load and resample instrument noise PSD from file.

        Parameters:
        -----------
        prepared_data_py : list
            List of dictionaries containing FFT data for each source (unused for file-based PSD)

        Returns:
        --------
        np.ndarray
            Array with shape (ky,kx,band,stokes) containing resampled noise PSD
        """
        print("Loading and resampling instrument noise model (PSD) via mean-pooling...")
        # Use the first band for single-band analysis
        band = self.bands[0]
        psd_filename = self.config.noise_psd_path.format(band=band.replace("GHz", ""))

        with fits.open(psd_filename) as hdul:
            psd_orig = hdul[0].data

        # Resample PSD to target resolution
        psd_resampled = self._resample_psd_to_target_resolution(psd_orig)

        # Create noise PSD array with shape (ky, kx, band, stokes)
        ny, nx = self.map_shape
        noise_psd_array = np.zeros((ny, nx, self.n_bands, 3), dtype=np.float32)

        # Fill the array for the single band (index 0)
        noise_psd_array[:, :, 0, 0] = psd_resampled  # T
        noise_psd_array[:, :, 0, 1] = psd_resampled * 2  # Q (2x noise)
        noise_psd_array[:, :, 0, 2] = psd_resampled * 2  # U (2x noise)

        return noise_psd_array

    def is_individual_psds(self):
        """Return False since this provides a global PSD for all sources."""
        return False

    def _resample_psd_to_target_resolution(self, psd_orig):
        """
        Resample PSD from original resolution to target resolution using mean-pooling.

        Parameters:
        -----------
        psd_orig : array_like
            Original PSD array

        Returns:
        --------
        array_like
            Resampled PSD array
        """
        orig_reso_arcmin = 0.25  # arcmin per pixel
        target_reso_arcmin = self.config.reso_arcmin
        target_shape = self.map_shape

        kmax_ratio = target_reso_arcmin / orig_reso_arcmin
        n_covered = int(target_shape[0] * kmax_ratio)
        n_inf = target_shape[0] - n_covered

        # Initialize with high value for unmeasured high-k modes
        psd_resampled = np.full(target_shape, 1e9)

        # Resample the low-k portion using mean-pooling
        lowk_psd = self._rebin_psd_with_averaging(psd_orig, (n_covered, n_covered))

        # Place resampled data in center
        start_idx = n_inf // 2
        end_idx = start_idx + n_covered
        psd_resampled[start_idx:end_idx, start_idx:end_idx] = np.fft.fftshift(lowk_psd)

        # Apply fftshift to get standard FFT layout
        psd_resampled = np.fft.fftshift(psd_resampled)

        # Set high values for k_x=0 modes to avoid division by zero
        psd_resampled[:, 0] = 1e12
        psd_resampled[:, -1] = 1e12

        return psd_resampled

    def _rebin_psd_with_averaging(self, psd_array, target_shape):
        """
        Rebin PSD array using averaging with optimized reshape/mean approach.

        Parameters:
        -----------
        psd_array : array_like
            Input PSD array
        target_shape : tuple
            Target shape (ny, nx)

        Returns:
        --------
        array_like
            Rebinned PSD array
        """
        old_ny, old_nx = psd_array.shape
        new_ny, new_nx = target_shape

        # Calculate bin sizes
        bin_y = old_ny // new_ny
        bin_x = old_nx // new_nx

        # Trim array to be evenly divisible by bin sizes
        trimmed_ny = bin_y * new_ny
        trimmed_nx = bin_x * new_nx
        trimmed_array = psd_array[:trimmed_ny, :trimmed_nx]

        # Reshape and average
        rebinned = trimmed_array.reshape(new_ny, bin_y, new_nx, bin_x).mean(axis=(1, 3))

        return rebinned


class KxAveragedIndividualCalculator(NoisePSDCalculator):
    """
    Calculate individual noise PSDs using k_x averaging with max heuristic.

    This implementation estimates noise by analyzing regions of each map that are
    away from the central source, then averages over k_y for each k_x mode and
    takes the element-wise maximum with the original PSD to avoid scattered low values.
    """

    def calculate_noise_psd(self, prepared_data_py):
        """
        Calculate individual noise PSDs for each source from the data.

        Parameters:
        -----------
        prepared_data_py : list
            List of dictionaries containing FFT data for each source

        Returns:
        --------
        list
            List of arrays with shape (ky,kx,band,stokes) containing noise PSDs for each source
        """
        print("Calculating individual data-driven noise PSDs for each source...")

        # Create noise mask with hole in center to avoid the source signal
        noise_mask = make_apod_mask_center_excised(self.map_shape, self.config.apodization_width_pix, self.config.noise_hole_radius_arcmin, self.config.reso_arcmin)

        noise_psds_list = []
        ny, nx = self.map_shape

        for i, data_fft in enumerate(prepared_data_py):
            print(f"  Processing source {i + 1}/{len(prepared_data_py)}")

            # Create array for this source: (ky, kx, band, stokes)
            source_noise_psd = np.zeros((ny, nx, self.n_bands, 3), dtype=np.float32)

            # Process each band and Stokes parameter
            for band_idx, band in enumerate(self.bands):
                for stokes_idx, stokes in enumerate(["T", "Q", "U"]):
                    # Extract data for this band and Stokes parameter
                    real_map = np.fft.ifft2(data_fft[band][stokes]).real

                    # Calculate individual noise PSD for this map
                    psd_2d = self._calculate_individual_noise_psd(real_map, noise_mask)

                    # Apply scaling for polarization
                    if stokes in ["Q", "U"]:
                        psd_2d *= 2.0  # Polarization has 2x the noise

                    source_noise_psd[:, :, band_idx, stokes_idx] = psd_2d

            noise_psds_list.append(source_noise_psd)

        print("Individual noise PSD calculation complete.")
        return noise_psds_list

    def is_individual_psds(self):
        """Return True since this provides individual PSDs per source."""
        return True

    def _calculate_individual_noise_psd(self, map_2d, noise_mask, sentinel_value=1e12):
        """
        Calculate noise PSD for a single map using k_y averaging.

        Parameters:
        -----------
        map_2d : array_like
            2D map to analyze
        noise_mask : array_like
            2D mask where 1 indicates regions to use for noise calculation
        sentinel_value : float
            Value to use for k_x=0 modes

        Returns:
        --------
        array_like
            2D noise PSD array
        """
        # Apply noise mask to isolate empty regions
        masked_map = map_2d * noise_mask

        # Take FFT and calculate power spectral density
        fft_2d = np.fft.fft2(masked_map)
        psd_2d = np.abs(fft_2d) ** 2

        # Normalize by the effective area (sum of mask squared)
        effective_area = np.sum(noise_mask**2)
        if effective_area > 0:
            psd_2d /= effective_area

        # Average over k_y for each k_x
        ny, nx = psd_2d.shape
        averaged_psd = np.zeros_like(psd_2d)

        for i in range(nx):
            # Average this column (constant k_x) over all k_y
            col_avg = np.mean(psd_2d[:, i])
            averaged_psd[:, i] = col_avg

        # Set k_x=0 modes to sentinel value to avoid division by zero
        averaged_psd[:, 0] = sentinel_value

        # Take element-wise maximum to avoid scattered low values (heuristic)
        psd = np.maximum(averaged_psd, psd_2d)

        return psd


class WhiteNoiseScaledCalculator(NoisePSDCalculator):
    """
    White noise assumption rescaled to center-excised standard deviation.

    This implementation assumes white noise but rescales the amplitude based on
    the empirical standard deviation calculated from center-excised regions of each map.
    """

    def calculate_noise_psd(self, prepared_data_py):
        """
        Calculate white noise PSDs rescaled to empirical standard deviations.

        Parameters:
        -----------
        prepared_data_py : list
            List of dictionaries containing FFT data for each source

        Returns:
        --------
        list
            List of arrays with shape (ky,kx,band,stokes) containing white noise PSDs for each source
        """
        print("Calculating white noise PSDs rescaled to empirical standard deviations...")

        # Create noise mask with hole in center to avoid the source signal
        noise_mask = make_apod_mask_center_excised(self.map_shape, self.config.apodization_width_pix, self.config.noise_hole_radius_arcmin, self.config.reso_arcmin)

        noise_psds_list = []
        ny, nx = self.map_shape

        for i, data_fft in enumerate(prepared_data_py):
            print(f"  Processing source {i + 1}/{len(prepared_data_py)}")

            # Create array for this source: (ky, kx, band, stokes)
            source_noise_psd = np.zeros((ny, nx, self.n_bands, 3), dtype=np.float32)

            # Process each band and Stokes parameter
            for band_idx, band in enumerate(self.bands):
                for stokes_idx, stokes in enumerate(["T", "Q", "U"]):
                    # Extract data for this band and Stokes parameter
                    real_map = np.fft.ifft2(data_fft[band][stokes]).real
                    masked_map = real_map * noise_mask
                    noise_power_level = np.mean(masked_map**2)
                    white_psd = np.full(self.map_shape, noise_power_level, dtype=np.float32)

                    # Set high values for k_x~0 modes to avoid division by zero
                    white_psd[:, 0] *= 100
                    white_psd[:, -1] *= 100
                    white_psd[0, 0] *= 100

                    source_noise_psd[:, :, band_idx, stokes_idx] = white_psd

            noise_psds_list.append(source_noise_psd)

        print("White noise PSD calculation complete.")
        return noise_psds_list

    def is_individual_psds(self):
        """Return True since this provides individual PSDs per source."""
        return True


class EnsembleAsdMeanCalculator(NoisePSDCalculator):
    """
    Calculate PSDs by averaging amplitude spectral densities across sources.

    This implementation takes the PSD of each source (with center-excised apodization),
    converts to amplitude spectral density (ASD), averages across all sources,
    then converts back to PSD.
    """

    def calculate_noise_psd(self, prepared_data_py):
        """
        Calculate ensemble-averaged ASD-derived PSDs.

        Parameters:
        -----------
        prepared_data_py : list
            List of dictionaries containing FFT data for each source

        Returns:
        --------
        np.ndarray
            Array with shape (ky,kx,band,stokes) containing ensemble-averaged PSD
        """
        print("Calculating ensemble-averaged ASD-derived PSDs...")

        # Create noise mask with hole in center to avoid the source signal
        noise_mask = make_apod_mask_center_excised(self.map_shape, self.config.apodization_width_pix, self.config.noise_hole_radius_arcmin, self.config.reso_arcmin)

        # Collect ASDs from all sources
        ny, nx = self.map_shape
        all_asds = np.zeros((len(prepared_data_py), ny, nx, self.n_bands, 3), dtype=np.float32)

        for i, data_fft in enumerate(prepared_data_py):
            print(f"  Processing source {i + 1}/{len(prepared_data_py)}")

            for band_idx, band in enumerate(self.bands):
                for stokes_idx, stokes in enumerate(["T", "Q", "U"]):
                    real_map = np.fft.ifft2(data_fft[band][stokes]).real

                    # Apply center-excised mask
                    masked_map = real_map * noise_mask

                    # Calculate PSD and convert to ASD
                    fft_2d = np.fft.fft2(masked_map)
                    psd_2d = np.abs(fft_2d) ** 2

                    # Normalize by effective area
                    effective_area = np.sum(noise_mask**2)
                    if effective_area > 0:
                        psd_2d /= effective_area

                    # Convert PSD to ASD (amplitude spectral density)
                    asd_2d = np.sqrt(psd_2d)
                    all_asds[i, :, :, band_idx, stokes_idx] = asd_2d

        # Average ASDs across all sources and convert back to PSD
        print("  Averaging ASDs across sources...")
        mean_asd = np.mean(all_asds, axis=0)  # Average over sources
        mean_psd = mean_asd**2

        # Apply scaling for polarization
        mean_psd[:, :, :, 1] *= 2.0  # Q polarization has 2x the noise
        mean_psd[:, :, :, 2] *= 2.0  # U polarization has 2x the noise

        print("Ensemble ASD averaging complete.")
        return mean_psd

    def is_individual_psds(self):
        """Return False since this provides a global ensemble-averaged PSD."""
        return False


class MultiBandCovarianceCalculator(NoisePSDCalculator):
    """
    Calculate multi-band covariance PSD for simultaneous fitting across frequency bands.

    This implementation creates a (ky,kx,band,band,stokes,stokes) covariance matrix capturing correlations
    between bands and Stokes parameters (T,Q,U), computed using center-excised apodization and averaged across sources.
    """

    def __init__(self, config, map_shape):
        super().__init__(config, map_shape)
        # Use the bands from config instead of hard-coding
        # They are already sorted by the parent class

    def calculate_noise_psd(self, prepared_data_py):
        """
        Calculate multi-band covariance PSD.

        Parameters:
        -----------
        prepared_data_py : list
            List of dictionaries containing FFT data for each source.
            Each dict has structure: {band: {stokes: fft_data}}

        Returns:
        --------
        np.ndarray
            Shape (ky,kx,band,band,stokes,stokes) covariance matrix
        """
        print("Calculating multi-band covariance PSD...")

        # Create noise mask with hole in center to avoid the source signal
        noise_mask = make_apod_mask_center_excised(self.map_shape, self.config.apodization_width_pix, self.config.noise_hole_radius_arcmin, self.config.reso_arcmin)

        ny, nx = self.map_shape
        stokes_params = ["T", "Q", "U"]
        n_bands = self.n_bands
        n_stokes = len(stokes_params)

        # Initialize covariance accumulator
        # Shape: (ky, kx, band, band, stokes, stokes)
        covariance_sum = np.zeros((ny, nx, n_bands, n_bands, n_stokes, n_stokes), dtype=np.complex64)
        n_sources = 0

        for i, data_dict in enumerate(prepared_data_py):
            print(f"  Processing source {i + 1}/{len(prepared_data_py)}")

            # Convert FFT data back to real space and apply noise mask
            masked_data = {}
            for band in self.bands:
                masked_data[band] = {}
                for stokes in stokes_params:
                    # Access the FFT data directly (it's already in Fourier space)
                    real_map = np.fft.ifft2(data_dict[band][stokes]).real
                    masked_map = real_map * noise_mask
                    # Go to Fourier space
                    masked_data[band][stokes] = np.fft.fft2(masked_map)

            # Calculate cross-covariances for this source
            for b1_idx, band1 in enumerate(self.bands):
                for s1_idx, stokes1 in enumerate(stokes_params):
                    for b2_idx, band2 in enumerate(self.bands):
                        for s2_idx, stokes2 in enumerate(stokes_params):
                            # Cross-PSD between (band1,stokes1) and (band2,stokes2)
                            cross_psd = masked_data[band1][stokes1] * np.conj(masked_data[band2][stokes2])
                            covariance_sum[:, :, b1_idx, b2_idx, s1_idx, s2_idx] += cross_psd

            n_sources += 1

        if n_sources == 0:
            raise RuntimeError("No valid sources found for multi-band covariance calculation")

        # Average across sources and normalize by effective area
        effective_area = np.sum(noise_mask**2)
        covariance_psd = covariance_sum / (n_sources * effective_area)

        print(f"Multi-band covariance calculation complete using {n_sources} sources.")
        return covariance_psd

    def is_individual_psds(self):
        """Return False since this provides a global covariance matrix."""
        return False


# Factory function to create appropriate noise PSD calculator
def create_noise_psd_calculator(config, map_shape):
    """
    Factory function to create the appropriate noise PSD calculator based on configuration.

    Parameters:
    -----------
    config : BeamFittingConfig
        Configuration object containing band and other parameters
    map_shape : tuple
        Shape of the maps (ny, nx)

    Returns:
    --------
    NoisePSDCalculator
        Appropriate noise PSD calculator instance
    """
    if config.noise_psd_method == "clusterfinder_psd":
        return ClusterfinderPSDCalculator(config, map_shape)
    elif config.noise_psd_method == "kx_averaged_individual":
        return KxAveragedIndividualCalculator(config, map_shape)
    elif config.noise_psd_method == "white_noise_scaled":
        return WhiteNoiseScaledCalculator(config, map_shape)
    elif config.noise_psd_method == "ensemble_asd_mean":
        return EnsembleAsdMeanCalculator(config, map_shape)
    elif config.noise_psd_method == "multiband_covariance":
        return MultiBandCovarianceCalculator(config, map_shape)
    else:
        raise ValueError(f"Unknown noise_psd_method: {config.noise_psd_method}")
