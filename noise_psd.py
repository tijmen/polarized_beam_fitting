"""
Noise Power Spectral Density (PSD) implementations for polarized beam fitting.

This module provides different approaches to estimating noise PSDs used in the
maximum-likelihood fitting of polarized beams.

There are several approaches for calculating the noise PSD. One of the major
differences is whether the noise PSD is fully diagonal (each ky,kx,band,stokes
is separate) or only diagonal in Fourier space (ky,ky independent, but band-band
and stokes-stokes off-diagonals).
We will use config.noise_psd_method to decide.
Currently, [clusterfinder_psd, kx_averaged, white_noise, ensemble_asd_mean, pca_psd, pca_psd_separate_tqu] are fully diagonal,
and [multiband_covariance] is only diagonal in Fourier space.
"""

from abc import ABC, abstractmethod

import numpy as np
from astropy.io import fits
from sklearn.decomposition import PCA

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
        self.n_bands = len(self.config.bands)
        self.map_shape = map_shape

    @abstractmethod
    def calculate_noise_psd(self, maps_numpy):
        """
        Calculate noise PSD(s) for the given data.

        Parameters:
        -----------
        maps_numpy : np.ndarray
            Array with shape (n_src, ny, nx, n_bands, 3) containing the source maps

        Returns:
        --------
        np.ndarray
            For single-band: (ky,kx,band,stokes) array for global PSD.
            For multi-band: (ky,kx,band,band,stokes,stokes) array for covariance.
        """
        pass


class ClusterfinderPSDCalculator(NoisePSDCalculator):
    """
    Load pre-computed instrument noise PSD from clusterfinder analysis.

    This implementation loads a noise PSD from a FITS file that was pre-computed
    from clusterfinder instrument characterization data, then resamples it to match
    the analysis map resolution using mean-pooling.
    """

    def calculate_noise_psd(self, maps_numpy):
        """
        Load and resample instrument noise PSD from file.

        Parameters:
        -----------
        maps_numpy : np.ndarray
            Array with shape (n_src, ny, nx, n_bands, 3) containing the source maps (unused for file-based PSD)

        Returns:
        --------
        np.ndarray
            Array with shape (ky,kx,band,stokes) containing resampled noise PSD
        """
        print("Loading and resampling instrument noise model (PSD) via mean-pooling...")
        # Use the first band for single-band analysis
        band = self.config.bands[0]
        psd_filename = self.config.noise_psd_path.format(band=band.replace("GHz", ""))

        with fits.open(psd_filename) as hdul:
            psd_orig = hdul[0].data

        # Resample PSD to target resolution
        psd_resampled = self._resample_psd_to_target_resolution(psd_orig)

        # Create noise PSD array with shape (ky, kx, band, stokes)
        ny, nx = self.map_shape
        noise_psd_array = np.zeros((ny, nx, self.n_bands, 3), dtype=self.config.dtype_np_real)

        # Fill the array for the single band (index 0)
        noise_psd_array[:, :, 0, 0] = psd_resampled  # T
        noise_psd_array[:, :, 0, 1] = psd_resampled * 2  # Q (2x noise)
        noise_psd_array[:, :, 0, 2] = psd_resampled * 2  # U (2x noise)

        return noise_psd_array

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
        ny, nx = self.map_shape

        # How much of the target Fourier grid is covered by the original PSD?
        # If target is finer (kmax_ratio > 1), coverage is a fraction 1/kmax_ratio of the target grid.
        # If target is coarser (kmax_ratio < 1), we can fill the whole target grid.
        kmax_ratio = orig_reso_arcmin / target_reso_arcmin

        # Covered side length on the target grid (isotropic)
        n_cov = int(round(ny / kmax_ratio)) if kmax_ratio > 1 else ny
        n_cov = max(1, min(ny, n_cov))

        # Downsample original PSD to the covered region size if needed
        if psd_orig.shape[0] >= n_cov and psd_orig.shape[1] >= n_cov:
            lowk_psd = self._rebin_psd_with_averaging(psd_orig, (n_cov, n_cov))
        else:
            # Fallback: gentle upsample if original is smaller (rare)
            from scipy.ndimage import zoom

            zoom_y = n_cov / psd_orig.shape[0]
            zoom_x = n_cov / psd_orig.shape[1]
            lowk_psd = zoom(psd_orig, (zoom_y, zoom_x), order=1)

        # Initialize with high value for unmeasured high-k modes
        psd_resampled = np.full((ny, nx), 1e9, dtype=self.config.dtype_np_real)

        # Place rebinned low-k box in the center of the target grid
        sy = (ny - n_cov) // 2
        sx = (nx - n_cov) // 2
        psd_resampled[sy : sy + n_cov, sx : sx + n_cov] = np.fft.fftshift(lowk_psd)

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

        assert new_ny <= old_ny, "_rebin_psd_with_averaging can only downsample, but is being asked to upsample"
        assert new_nx <= old_nx, "_rebin_psd_with_averaging can only downsample, but is being asked to upsample"

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


class KxAveragedCalculator(NoisePSDCalculator):
    """
    Calculate individual noise PSDs using k_x averaging with max heuristic,
    then average over all sources.

    This implementation estimates noise by analyzing regions of each map that are
    away from the central source, then averages over k_y for each k_x mode and
    takes the element-wise maximum with the original PSD to avoid scattered low values.
    """

    def calculate_noise_psd(self, maps_numpy):
        """
        Calculate individual noise PSDs for each source from the data.

        Parameters:
        -----------
        maps_numpy : np.ndarray
            Array with shape (n_src, ny, nx, n_bands, 3) containing the source maps

        Returns:
        --------
        np.ndarray
            Array with shape (ny, nx, band, stokes) containing global noise PSD
        """
        print("Calculating individual data-driven noise PSDs for each source...")

        # Create noise mask with hole in center to avoid the source signal
        noise_mask = make_apod_mask_center_excised(
            self.map_shape,
            self.config.apodization_width_pix,
            self.config.noise_hole_radius_arcmin,
            self.config.reso_arcmin,
        )

        noise_psds_list = []
        ny, nx = self.map_shape
        n_src = maps_numpy.shape[0]

        for i in range(n_src):
            print(f"  Processing source {i + 1}/{n_src}")

            # Create array for this source: (ky, kx, band, stokes)
            source_noise_psd = np.zeros((ny, nx, self.n_bands, 3), dtype=self.config.dtype_np_real)

            # Process each band and Stokes parameter
            for band_idx, band in enumerate(self.config.bands):
                for stokes_idx, stokes in enumerate(["T", "Q", "U"]):
                    # Extract data for this band and Stokes parameter
                    real_map = maps_numpy[i, :, :, band_idx, stokes_idx]

                    # Calculate individual noise PSD for this map
                    psd_2d = self._calculate_individual_noise_psd(real_map, noise_mask)

                    # Apply scaling for polarization
                    if stokes in ["Q", "U"]:
                        psd_2d *= 2.0  # Polarization has 2x the noise

                    source_noise_psd[:, :, band_idx, stokes_idx] = psd_2d

            noise_psds_list.append(source_noise_psd)

        # Average the PSDs across all sources
        noise_psds = np.array(noise_psds_list)
        global_noise_psd = np.mean(noise_psds, axis=0)

        print("Noise PSD calculation complete.")
        return global_noise_psd

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


class EnsembleAsdMeanCalculator(NoisePSDCalculator):
    """
    Calculate PSDs by averaging amplitude spectral densities across sources.

    This implementation takes the PSD of each source (with center-excised apodization),
    converts to amplitude spectral density (ASD), averages across all sources,
    then converts back to PSD.
    """

    def calculate_noise_psd(self, maps_numpy):
        """
        Calculate ensemble-averaged ASD-derived PSDs.

        Parameters:
        -----------
        maps_numpy : np.ndarray
            Array with shape (n_src, ny, nx, n_bands, 3) containing the source maps

        Returns:
        --------
        np.ndarray
            Array with shape (ky,kx,band,stokes) containing ensemble-averaged PSD
        """
        print("Calculating ensemble-averaged ASD-derived PSDs...")

        # Create noise mask with hole in center to avoid the source signal
        noise_mask = make_apod_mask_center_excised(
            self.map_shape,
            self.config.apodization_width_pix,
            self.config.noise_hole_radius_arcmin,
            self.config.reso_arcmin,
        )

        # Collect ASDs from all sources
        ny, nx = self.map_shape
        n_src = maps_numpy.shape[0]
        all_asds = np.zeros((n_src, ny, nx, self.n_bands, 3), dtype=self.config.dtype_np_real)

        for i in range(n_src):
            print(f"  Processing source {i + 1}/{n_src}")

            for band_idx, band in enumerate(self.config.bands):
                for stokes_idx, stokes in enumerate(["T", "Q", "U"]):
                    real_map = maps_numpy[i, :, :, band_idx, stokes_idx]

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

    def calculate_noise_psd(self, maps_numpy):
        """
        Calculate multi-band covariance PSD.

        Parameters:
        -----------
        maps_numpy : np.ndarray
            Array with shape (n_src, ky, kx, n_bands, n_stokes)
            Contains the source maps.

        Returns:
        --------
        np.ndarray
            Shape (ky, kx, band, band, stokes, stokes) covariance matrix
        """
        print("Calculating multi-band covariance PSD...")

        noise_mask = make_apod_mask_center_excised(
            self.map_shape,
            self.config.apodization_width_pix,
            self.config.noise_hole_radius_arcmin,
            self.config.reso_arcmin,
        )  # shape (ky, kx)
        # apply mask over spatial dims
        masked_maps = maps_numpy * noise_mask[None, :, :, None, None]
        # FFT in spatial dimensions
        masked_maps_fft = np.fft.fft2(masked_maps, axes=(1, 2))  # (n_src, ky, kx, band, stokes)

        # Source-averaged cross-PSD over band and stokes
        n_src = masked_maps_fft.shape[0]
        covariance_sum = np.einsum("nyxbs,nyxct->yxbcst", masked_maps_fft, np.conj(masked_maps_fft))  # (ky, kx, band, band, stokes, stokes)

        effective_area = np.sum(noise_mask**2)
        covariance_psd = covariance_sum / (n_src * effective_area)

        print(f"Multi-band covariance calculation complete using {n_src} sources.")
        return covariance_psd


class PcaPsdCalculator(NoisePSDCalculator):
    """

    PcaPsdCalculator is a per-source PSD calculator holding whatever PSD I'm currently developing. It is the default.

    Calculates noise PSDs using a data-driven, log-space PCA model. Each component is a PSD. Log space is good
    because the individual PSDs have chi-squared-distributed noise with two degrees of freedom. I believe the
    log of that will have Gaussian noise.
    """

    def calculate_noise_psd(self, maps_numpy: np.ndarray) -> np.ndarray:
        """
        Calculates noise PSDs using a log-space PCA model built from the data.

        Parameters:
        -----------
        maps_numpy : np.ndarray
            Array with shape (n_src, ny, nx, n_bands, n_stokes) containing the maps.

        Returns:
        --------
        np.ndarray
            Array of the same shape containing the reconstructed 2D noise PSDs.
        """
        print(f"Calculating noise PSDs with log-space PCA model ({self.config.n_pca_components} components)...")
        n_src, ny, nx, n_bands, n_stokes = maps_numpy.shape
        map_shape = (ny, nx)

        noise_mask = make_apod_mask_center_excised(
            map_shape,
            self.config.apodization_width_pix,
            self.config.noise_hole_radius_arcmin,
            self.config.reso_arcmin,
        )
        effective_area = np.sum(noise_mask**2)
        print("Collating all map PSDs...")
        all_psds_flat_linear = []
        for i in range(n_src):
            for band_idx in range(n_bands):
                for stokes_idx in range(n_stokes):
                    real_map = maps_numpy[i, :, :, band_idx, stokes_idx]
                    masked_map = real_map * noise_mask
                    fft_2d = np.fft.fft2(masked_map)
                    psd_2d = np.abs(fft_2d) ** 2 / effective_area
                    all_psds_flat_linear.append(psd_2d.flatten())

        X_linear = np.array(all_psds_flat_linear)
        X_log = np.log(X_linear)
        print("Performing PCA on log-transformed PSDs...")
        mean_log_psd = np.mean(X_log, axis=0)
        X_log_centered = X_log - mean_log_psd
        pca = PCA(n_components=self.config.n_pca_components, svd_solver="randomized", random_state=42)
        pca.fit(X_log_centered)
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

        print("Reconstructing denoised PSDs...")
        coeffs = pca.transform(X_log_centered)
        X_reconstructed_log_centered = pca.inverse_transform(coeffs)
        X_reconstructed_log = X_reconstructed_log_centered + mean_log_psd
        X_reconstructed_linear = np.exp(X_reconstructed_log)

        per_source_psd_array = np.zeros_like(maps_numpy, dtype=self.config.dtype_np_real)
        map_idx = 0
        for i in range(n_src):
            for band_idx in range(n_bands):
                for stokes_idx in range(n_stokes):
                    per_source_psd_array[i, :, :, band_idx, stokes_idx] = X_reconstructed_linear[map_idx].reshape(map_shape)
                    map_idx += 1

        print("PCA-based PSD calculation complete.")
        return per_source_psd_array


class PcaPsdSeparateTQUCalculator(NoisePSDCalculator):
    """
    PcaPsdSeparateTQUCalculator performs separate PCA analyses: one on temperature (T) only,
    and another on the combination of polarization (Q,U) maps.

    This approach allows for different noise structures between temperature and polarization
    measurements, which can be important for cosmic microwave background observations.
    """

    def calculate_noise_psd(self, maps_numpy: np.ndarray) -> np.ndarray:
        """
        Calculates noise PSDs using separate log-space PCA models for T and for (Q,U).

        Parameters:
        -----------
        maps_numpy : np.ndarray
            Array with shape (n_src, ny, nx, n_bands, n_stokes) containing the maps.

        Returns:
        --------
        np.ndarray
            Array of the same shape containing the reconstructed 2D noise PSDs.
        """
        print(
            f"Calculating noise PSDs with separate PCA models (T: {self.config.n_pca_components}, Q,U: {self.config.n_pca_components} components)..."
        )
        n_src, ny, nx, n_bands, n_stokes = maps_numpy.shape
        map_shape = (ny, nx)

        noise_mask = make_apod_mask_center_excised(
            map_shape,
            self.config.apodization_width_pix,
            self.config.noise_hole_radius_arcmin,
            self.config.reso_arcmin,
        )
        effective_area = np.sum(noise_mask**2)

        # Initialize output array
        per_source_psd_array = np.zeros_like(maps_numpy, dtype=self.config.dtype_np_real)

        # Process T maps (stokes_idx = 0) separately
        print("Collating T (temperature) map PSDs...")
        all_psds_T_flat_linear = []
        for i in range(n_src):
            for band_idx in range(n_bands):
                real_map = maps_numpy[i, :, :, band_idx, 0]  # T component
                masked_map = real_map * noise_mask
                fft_2d = np.fft.fft2(masked_map)
                psd_2d = np.abs(fft_2d) ** 2 / effective_area
                all_psds_T_flat_linear.append(psd_2d.flatten())

        X_T_linear = np.array(all_psds_T_flat_linear)
        X_T_log = np.log(X_T_linear)
        print("Performing PCA on T log-transformed PSDs...")
        mean_log_psd_T = np.mean(X_T_log, axis=0)
        X_T_log_centered = X_T_log - mean_log_psd_T
        pca_T = PCA(n_components=self.config.n_pca_components, svd_solver="randomized", random_state=42)
        pca_T.fit(X_T_log_centered)
        print(f"T PCA explained variance ratio: {pca_T.explained_variance_ratio_}")

        print("Reconstructing denoised T PSDs...")
        coeffs_T = pca_T.transform(X_T_log_centered)
        X_T_reconstructed_log_centered = pca_T.inverse_transform(coeffs_T)
        X_T_reconstructed_log = X_T_reconstructed_log_centered + mean_log_psd_T
        X_T_reconstructed_linear = np.exp(X_T_reconstructed_log)

        # Fill T component in output array
        map_idx = 0
        for i in range(n_src):
            for band_idx in range(n_bands):
                per_source_psd_array[i, :, :, band_idx, 0] = X_T_reconstructed_linear[map_idx].reshape(map_shape)
                map_idx += 1

        # Process Q,U maps (stokes_idx = 1,2) together
        print("Collating Q,U (polarization) map PSDs...")
        all_psds_QU_flat_linear = []
        for i in range(n_src):
            for band_idx in range(n_bands):
                for stokes_idx in [1, 2]:  # Q and U components
                    real_map = maps_numpy[i, :, :, band_idx, stokes_idx]
                    masked_map = real_map * noise_mask
                    fft_2d = np.fft.fft2(masked_map)
                    psd_2d = np.abs(fft_2d) ** 2 / effective_area
                    all_psds_QU_flat_linear.append(psd_2d.flatten())

        X_QU_linear = np.array(all_psds_QU_flat_linear)
        X_QU_log = np.log(X_QU_linear)
        print("Performing PCA on Q,U log-transformed PSDs...")
        mean_log_psd_QU = np.mean(X_QU_log, axis=0)
        X_QU_log_centered = X_QU_log - mean_log_psd_QU
        pca_QU = PCA(n_components=self.config.n_pca_components, svd_solver="randomized", random_state=42)
        pca_QU.fit(X_QU_log_centered)
        print(f"Q,U PCA explained variance ratio: {pca_QU.explained_variance_ratio_}")

        print("Reconstructing denoised Q,U PSDs...")
        coeffs_QU = pca_QU.transform(X_QU_log_centered)
        X_QU_reconstructed_log_centered = pca_QU.inverse_transform(coeffs_QU)
        X_QU_reconstructed_log = X_QU_reconstructed_log_centered + mean_log_psd_QU
        X_QU_reconstructed_linear = np.exp(X_QU_reconstructed_log)

        # Fill Q,U components in output array
        map_idx = 0
        for i in range(n_src):
            for band_idx in range(n_bands):
                for stokes_idx in [1, 2]:  # Q and U components
                    per_source_psd_array[i, :, :, band_idx, stokes_idx] = X_QU_reconstructed_linear[map_idx].reshape(map_shape)
                    map_idx += 1

        print("Separate T and Q,U PCA-based PSD calculation complete.")
        return per_source_psd_array


class WhiteNoiseCalculator(NoisePSDCalculator):
    """
    Calculate simple white noise PSD with constant values.

    This implementation assumes white noise with constant PSD values
    across all k-space for testing and baseline comparisons.
    """

    def calculate_noise_psd(self, maps_numpy):
        """
        Calculate simple white noise PSD with constant values.

        Parameters:
        -----------
        maps_numpy : np.ndarray
            Array with shape (n_src, ny, nx, n_bands, 3) containing the source maps

        Returns:
        --------
        np.ndarray
            Array with shape (ky,kx,band,stokes) containing white noise PSD
        """
        print("Generating simple white noise PSD...")

        ny, nx = self.map_shape

        # Create white noise PSD array with all ones
        noise_psd_array = np.ones((ny, nx, self.n_bands, 3), dtype=self.config.dtype_np_real)

        # Apply scaling for polarization
        noise_psd_array[:, :, :, 1] *= 2.0  # Q polarization has 2x the noise
        noise_psd_array[:, :, :, 2] *= 2.0  # U polarization has 2x the noise

        print("White noise PSD complete.")
        return noise_psd_array


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
    elif config.noise_psd_method == "kx_averaged":
        return KxAveragedCalculator(config, map_shape)
    elif config.noise_psd_method == "ensemble_asd_mean":
        return EnsembleAsdMeanCalculator(config, map_shape)
    elif config.noise_psd_method == "multiband_covariance":
        return MultiBandCovarianceCalculator(config, map_shape)
    elif config.noise_psd_method == "pca_psd":
        return PcaPsdCalculator(config, map_shape)
    elif config.noise_psd_method == "pca_psd_separate_tqu":
        return PcaPsdSeparateTQUCalculator(config, map_shape)
    elif config.noise_psd_method == "white_noise":
        return WhiteNoiseCalculator(config, map_shape)
    else:
        raise ValueError(f"Unknown noise_psd_method: {config.noise_psd_method}")
