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

import pickle
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

        # Reorder to interleave band and Stokes axes for downstream reshapes
        covariance_psd = np.transpose(covariance_psd, (0, 1, 2, 4, 3, 5))

        print(f"Multi-band covariance calculation complete using {n_src} sources.")
        return covariance_psd


class PcaMultiBandCalculator(NoisePSDCalculator):
    """PCA-regularized multi-band precision estimator."""

    def calculate_noise_psd(self, maps_numpy: np.ndarray) -> np.ndarray:
        print(f"Calculating PCA-based multi-band precision ({self.config.n_pca_components} components)...")

        n_src, ny, nx, n_bands, n_stokes = maps_numpy.shape
        n_dim = n_bands * n_stokes

        noise_mask = make_apod_mask_center_excised(
            self.map_shape,
            self.config.apodization_width_pix,
            self.config.noise_hole_radius_arcmin,
            self.config.reso_arcmin,
        )

        masked_maps = maps_numpy * noise_mask[None, :, :, None, None]
        masked_maps_fft = np.fft.fft2(masked_maps, axes=(1, 2))

        effective_area = np.sum(noise_mask**2)
        if effective_area <= 0:
            raise ValueError("Effective area of noise mask must be positive.")
        masked_maps_fft = masked_maps_fft / np.sqrt(effective_area)

        fft_reshaped = masked_maps_fft.reshape(n_src, -1)
        n_features = fft_reshaped.shape[1]

        X_real = np.hstack([fft_reshaped.real, fft_reshaped.imag])

        n_components = min(self.config.n_pca_components, n_src - 1)
        if n_components <= 0:
            raise ValueError("n_pca_components must be positive and strictly less than the number of sources for PCA precision estimation.")

        print(f"  Performing PCA with {n_components} components...")
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
        pca.fit(X_real)

        print(f"  PCA explained variance ratio: {pca.explained_variance_ratio_}")
        total_var_top = np.sum(pca.explained_variance_)
        print(f"  Total variance captured: {total_var_top:.4g}")

        total_data_variance = np.var(X_real)
        dof_total = X_real.shape[1]
        dof_residual = max(dof_total - n_components, 1)
        variance_floor = (total_data_variance * dof_total - total_var_top) / dof_residual

        if variance_floor <= 0:
            variance_floor = 1e-9 * max(total_data_variance, 1.0)
            print(f"  Warning: variance floor non-positive; using fallback {variance_floor:.2e}")
        else:
            print(f"  Estimated variance floor: {variance_floor:.4g}")

        components_real = pca.components_[:, :n_features]
        components_imag = pca.components_[:, n_features:]
        components_complex = (
            (components_real + 1j * components_imag).reshape(n_components, ny, nx, n_bands, n_stokes).astype(self.config.dtype_np_complex)
        )

        eigenvalues = pca.explained_variance_ * n_src
        max_eig = np.max(eigenvalues) if eigenvalues.size else 0.0
        denom_clip = 1e-12 * max(max_eig, 1.0)
        precision_eigs = 1.0 / np.maximum(eigenvalues, denom_clip)
        precision_floor = 1.0 / (variance_floor * n_src)
        weights = precision_eigs - precision_floor

        precision_low_rank = np.einsum(
            "c,cyxbs,cyxBT->yxbsBT",
            weights,
            components_complex,
            np.conj(components_complex),
        )

        identity = np.eye(n_dim, dtype=self.config.dtype_np_complex).reshape(n_bands, n_stokes, n_bands, n_stokes)
        precision_matrix = precision_low_rank + precision_floor * identity[None, None, :, :, :, :]
        precision_matrix = 0.5 * (precision_matrix + np.swapaxes(precision_matrix, 2, 4).swapaxes(3, 5).conj())

        print("PCA-based multi-band precision calculation complete.")
        return precision_matrix


class ParametricPrefitCalculator(NoisePSDCalculator):
    """Load a pre-fit precision matrix model from disk."""

    def __init__(self, config, map_shape):
        super().__init__(config, map_shape)
        path = getattr(self.config, "parametric_prefit_precision_path", None)
        if not path:
            raise ValueError("parametric_prefit_precision_path must be set to use the 'parametric_prefit' noise PSD method.")

        with open(path, "rb") as handle:
            payload = pickle.load(handle)

        if "precision" not in payload:
            raise ValueError(f"Prefit precision file '{path}' is missing the 'precision' entry.")

        precision_stack = np.asarray(payload["precision"])
        if precision_stack.ndim != 7:
            raise ValueError(
                f"Expected 7D precision tensor (n_samples, ny, nx, band, stokes, band, stokes); received shape {precision_stack.shape}."
            )

        ny_file, nx_file = precision_stack.shape[1:3]
        if (ny_file, nx_file) != map_shape:
            raise ValueError(
                f"Prefit precision grid is {ny_file}x{nx_file} but fitter maps use {map_shape[0]}x{map_shape[1]}; resampling is not supported yet."
            )

        n_stokes = precision_stack.shape[4]
        if n_stokes != 3:
            raise ValueError(f"Prefit precision tensor expects 3 Stokes components, found axis length {n_stokes}.")

        if precision_stack.shape[3] != self.n_bands or precision_stack.shape[5] != self.n_bands:
            raise ValueError(
                "Prefit precision tensor band axes do not match configuration bands; regenerate the parametric model for the current band list."
            )

        self._precision = precision_stack.mean(axis=0).astype(self.config.dtype_np_complex)
        self._metadata = payload
        self.ell_max_prefit = float(payload.get("ell_max")) if payload.get("ell_max") is not None else None

    def calculate_noise_psd(self, maps_numpy):
        _ = maps_numpy  # Unused; all information lives in the pre-fit tensor.
        return self._precision


class PcaPsdSeparateTQUCalculator(NoisePSDCalculator):
    """
    PcaPsdSeparateTQUCalculator performs separate PCA analyses for each Stokes parameter:
    one for temperature (T), one for Q polarization, and one for U polarization.

    This approach allows for different noise structures between temperature and each
    polarization component, which can be important for cosmic microwave background observations.
    """

    def calculate_noise_psd(self, maps_numpy: np.ndarray) -> np.ndarray:
        """
        Calculates noise PSDs using separate log-space PCA models for T, Q, and U.

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
            f"Calculating noise PSDs with separate PCA models (T: {self.config.n_pca_components}, Q: {self.config.n_pca_components}, U: {self.config.n_pca_components} components)..."
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

        # Process each Stokes parameter separately (T, Q, U)
        stokes_names = ["T", "Q", "U"]
        for stokes_idx, stokes_name in enumerate(stokes_names):
            print(f"Collating {stokes_name} map PSDs...")
            all_psds_flat_linear = []
            for i in range(n_src):
                for band_idx in range(n_bands):
                    real_map = maps_numpy[i, :, :, band_idx, stokes_idx]
                    masked_map = real_map * noise_mask
                    fft_2d = np.fft.fft2(masked_map)
                    psd_2d = np.abs(fft_2d) ** 2 / effective_area
                    all_psds_flat_linear.append(psd_2d.flatten())

            X_linear = np.array(all_psds_flat_linear)
            X_log = np.log(X_linear)
            print(f"Performing PCA on {stokes_name} log-transformed PSDs...")
            mean_log_psd = np.mean(X_log, axis=0)
            X_log_centered = X_log - mean_log_psd
            pca = PCA(n_components=self.config.n_pca_components, svd_solver="randomized", random_state=42)
            pca.fit(X_log_centered)
            print(f"{stokes_name} PCA explained variance ratio: {pca.explained_variance_ratio_}")

            print(f"Reconstructing denoised {stokes_name} PSDs...")
            coeffs = pca.transform(X_log_centered)
            X_reconstructed_log_centered = pca.inverse_transform(coeffs)
            X_reconstructed_log = X_reconstructed_log_centered + mean_log_psd
            X_reconstructed_linear = np.exp(X_reconstructed_log)

            # Fill this Stokes component in output array
            map_idx = 0
            for i in range(n_src):
                for band_idx in range(n_bands):
                    per_source_psd_array[i, :, :, band_idx, stokes_idx] = X_reconstructed_linear[map_idx].reshape(map_shape)
                    map_idx += 1

        print("Separate T, Q, and U PCA-based PSD calculation complete.")
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
    elif config.noise_psd_method == "pca_multiband_covariance":
        return PcaMultiBandCalculator(config, map_shape)
    elif config.noise_psd_method == "parametric_prefit":
        return ParametricPrefitCalculator(config, map_shape)
    elif config.noise_psd_method == "pca_psd_separate_tqu":
        return PcaPsdSeparateTQUCalculator(config, map_shape)
    elif config.noise_psd_method == "white_noise":
        return WhiteNoiseCalculator(config, map_shape)
    else:
        raise ValueError(f"Unknown noise_psd_method: {config.noise_psd_method}")
