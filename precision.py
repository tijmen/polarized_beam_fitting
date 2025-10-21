"""
Fourier-domain covariance and precision estimators for polarized beam fitting.

Each estimator corresponds to a choice of ``config.covariance_method`` and
returns a model for the covariance or its inverse (precision) on the discrete
Fourier grid defined by the map geometry.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import camb
import numpy as np
from astropy.io import fits
from sklearn.decomposition import PCA

from .utils import (
    compute_fourier_frequency_axes,
    compute_rectangular_ell_cut_indices,
    make_apod_mask_center_excised,
    make_apodization_mask,
)

CMB_SUPERSAMPLING = 4
COVARIANCE_EIGEN_EPS = 1e-6

# Tcal https://sptlocal.grid.uchicago.edu/~yomori/20192020_lensing/Tcal/v3/spt3g20192020_tcal.html#updated-calibration
# Pcal https://pole.uchicago.edu/spt3g/index.php/File:20231009_EETETT_Updates.pdf
CMB_CALIBRATION_FACTORS = {
    "T": {"90GHz": 1.07, "150GHz": 1.02, "220GHz": 1.01},
    "Q": {"90GHz": 1.05, "150GHz": 1.06, "220GHz": 1.17},
    "U": {"90GHz": 1.05, "150GHz": 1.06, "220GHz": 1.17},
}


def _band_suffix(band_name: str) -> str:
    """Return a compact suffix (e.g. '150') extracted from a band label such as '150GHz'."""
    digits = "".join(ch for ch in band_name if ch.isdigit())
    return digits or band_name


def _compute_rms_table(covariance: np.ndarray, config) -> List[Tuple[str, float]]:
    """
    Return (label, rms_uk_arcmin) pairs for each band/stokes diagonal extracted from the provided covariance grid.

    Parameters
    ----------
    covariance : np.ndarray
        Covariance grid with axes (..., band, stokes, band, stokes).
    reso_arcmin : float
        Map resolution in arcminutes.
    bands : list[str]
        Ordered list of band labels matching the covariance dimensions.
    """
    rms_entries: List[Tuple[str, float]] = []
    diag_view = np.asarray(covariance).real
    stokes_labels = "TQU"

    for band_idx, band in enumerate(config.bands):
        suffix = _band_suffix(band)
        for stokes_idx, st_label in enumerate(stokes_labels):
            diag_vals = diag_view[..., band_idx, stokes_idx, band_idx, stokes_idx]
            diag_vals = np.clip(diag_vals, 0.0, None)
            rms_mK = float(np.sqrt(np.mean(diag_vals)))
            rms_uk_arcmin = rms_mK / config.map_size_pix * 1000.0 * config.reso_arcmin
            rms_entries.append((f"{st_label}{suffix}", rms_uk_arcmin))
    return rms_entries


def _print_rms_summary(title: str, covariance: np.ndarray, config) -> None:
    """Log RMS summaries (in µK-arcmin) for each band/stokes diagonal of the provided covariance."""
    entries = _compute_rms_table(covariance, config)
    if not entries:
        return
    formatted = ", ".join(f"{label}={value:.2f} µK-arcmin" for label, value in entries)
    print(f"{title}: {formatted}")


def _ensure_fft_cut_indices(map_shape: Tuple[int, int], config) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return Fourier indices consistent with the configured ellmax cutoff."""
    idx_y, idx_x = compute_rectangular_ell_cut_indices(map_shape, config.reso_arcmin, getattr(config, "ellmax", None))
    if idx_y is None or idx_x is None:
        return None, None
    return idx_y, idx_x


def _smooth_highpass_1d(ell_vals: np.ndarray, ell0: float = 360.0, ell1: float = 720.0) -> np.ndarray:
    """Raised-cosine high-pass filter used to suppress large-scale CMB power."""
    print(f"Creating high-pass filter turning on between {ell0} and {ell1}...")
    hp_filter = np.ones_like(ell_vals, dtype=float)
    hp_filter[ell_vals <= ell0] = 0.0
    transition = (ell_vals > ell0) & (ell_vals < ell1)
    if np.any(transition):
        frac = (ell_vals[transition] - ell0) / (ell1 - ell0)
        hp_filter[transition] = 0.5 * (1.0 - np.cos(np.pi * frac))
    return hp_filter


def _project_to_spd(matrix: np.ndarray, eps: float = COVARIANCE_EIGEN_EPS) -> np.ndarray:
    """Project a covariance matrix to the nearest symmetric positive-definite matrix."""
    symm = 0.5 * (matrix + matrix.T)
    diag = np.diag(symm)
    positive_diag = diag[diag > 0]
    assert positive_diag.size > 0, "No positive diagonal elements found in covariance matrix"
    scale = np.median(positive_diag)
    floor = eps * scale
    eigvals, eigvecs = np.linalg.eigh(symm)
    if np.any(eigvals < floor):
        print(f"Warning: {np.sum(eigvals < floor)} small or negative eigenvalues found in covariance matrix. Setting to {floor:.2e}")
    eigvals = np.maximum(eigvals, floor)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _compute_cmb_covariance(config, ny: int, nx: int) -> np.ndarray:
    """
    Build the CMB covariance grid in T/Q/U and apply calibrations to expand to
    full band-band-stokes-stokes covariance.

    Input:
    ------
    config: Config
        Configuration object containing the map shape and resolution.
    ny: int
        Number of pixels remaining in the Fourier domain after applying the Fourier cut indices.
    nx: int
        Number of pixels remaining in the Fourier domain after applying the Fourier cut indices.

    Output:
    -------
    cov_cmb: np.ndarray
        CMB covariance grid
    """
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.1200, mnu=0.06, omk=0, tau=0.0544)
    pars.InitPower.set_params(As=2.1e-9, ns=0.9649)
    ell_max_cmb = 5000
    pars.set_for_lmax(ell_max_cmb, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)
    ell_camb = np.arange(powers["total"].shape[0])
    cl_tt_raw = powers["total"][:, 0] * 1e-6  # convert from µK^2 to mK^2
    cl_ee_raw = powers["total"][:, 1] * 1e-6
    cl_bb_raw = powers["total"][:, 2] * 1e-6
    cl_te_raw = powers["total"][:, 3] * 1e-6

    hp_filter = _smooth_highpass_1d(ell_camb)
    cl_tt = cl_tt_raw * hp_filter
    cl_ee = cl_ee_raw * hp_filter
    cl_bb = cl_bb_raw * hp_filter
    cl_te = cl_te_raw * hp_filter

    ss = CMB_SUPERSAMPLING
    ny_highres = ny * ss  # side length of the supersampled Fourier domain
    nx_highres = nx * ss
    dtheta_rad = config.reso_arcmin * (np.pi / (180.0 * 60.0))  # map resolution in radians
    map_length_rad = config.map_size_pix * dtheta_rad  # radians
    # map_area_sr = map_length_rad**2  # steradians
    pixel_area_sr = dtheta_rad**2

    ell_x_highres = np.fft.fftfreq(nx_highres)
    ell_x_highres /= ell_x_highres[1] - ell_x_highres[0]
    ell_x_highres *= 2 * np.pi / (map_length_rad * ss)
    ell_y_highres = np.fft.fftfreq(ny_highres)
    ell_y_highres /= ell_y_highres[1] - ell_y_highres[0]
    ell_y_highres *= 2 * np.pi / (map_length_rad * ss)
    ell_x_grid_highres, ell_y_grid_highres = np.meshgrid(ell_x_highres, ell_y_highres, indexing="xy")
    ell_radial_highres = np.sqrt(ell_x_grid_highres**2 + ell_y_grid_highres**2)

    def interp(cls):
        return np.interp(ell_radial_highres.ravel(), ell_camb, cls, left=0.0, right=0.0).reshape(ell_radial_highres.shape)

    cl_tt_grid = interp(cl_tt)
    cl_ee_grid = interp(cl_ee)
    cl_bb_grid = interp(cl_bb)
    cl_te_grid = interp(cl_te)

    phi = np.arctan2(ell_y_grid_highres, ell_x_grid_highres)
    c2phi = np.cos(2.0 * phi)
    s2phi = np.sin(2.0 * phi)

    cov_tqu_highres = np.zeros((ny_highres, nx_highres, 3, 3), dtype=config.dtype_np_real)
    cov_tqu_highres[..., 0, 0] = cl_tt_grid
    cov_tqu_highres[..., 0, 1] = cov_tqu_highres[..., 1, 0] = cl_te_grid * c2phi
    cov_tqu_highres[..., 0, 2] = cov_tqu_highres[..., 2, 0] = cl_te_grid * s2phi
    cov_tqu_highres[..., 1, 1] = cl_ee_grid * c2phi**2 + cl_bb_grid * s2phi**2
    cov_tqu_highres[..., 2, 2] = cl_ee_grid * s2phi**2 + cl_bb_grid * c2phi**2
    cov_tqu_highres[..., 1, 2] = cov_tqu_highres[..., 2, 1] = (cl_ee_grid - cl_bb_grid) * s2phi * c2phi

    cov_tqu_highres_shifted = np.fft.fftshift(cov_tqu_highres)
    cov_tqu_highres_shifted = 0.5 * np.roll(cov_tqu_highres_shifted, (-1, -1), axis=(0, 1)) + 0.5 * cov_tqu_highres_shifted
    cov_tqu_shifted = cov_tqu_highres_shifted.reshape(ny, ss, nx, ss, 3, 3).sum(axis=(1, 3))
    cov_tqu = np.fft.ifftshift(cov_tqu_shifted)

    apod_mask = make_apodization_mask((ny, nx), config.apodization_width_pix)
    apod_mask_fft = np.fft.fft2(apod_mask)
    window_full = np.abs(apod_mask_fft) ** 2 / (ny * nx)  # total area slightly < 1
    inds_y = np.concatenate((np.arange(0, ny // 2 + 1), np.arange(-ny // 2 + 1, 0)))
    inds_x = np.concatenate((np.arange(0, nx // 2 + 1), np.arange(-nx // 2 + 1, 0)))
    window = window_full[np.ix_(inds_y, inds_x)]
    fftwindow = np.fft.fft2(window)
    fft_cov_tqu = np.fft.fft2(cov_tqu, axes=(0, 1))
    cov_tqu_convolved = np.fft.ifft2(fft_cov_tqu * fftwindow[:, :, None, None], axes=(0, 1)).real

    n_bands = len(config.bands)
    cov_cmb = np.zeros((ny, nx, n_bands, 3, n_bands, 3), dtype=config.dtype_np_real)
    for band_i_idx, band_i_label in enumerate(config.bands):
        for stokes_i_idx, stokes_i_label in enumerate(["T", "Q", "U"]):
            for band_j_idx, band_j_label in enumerate(config.bands):
                for stokes_j_idx, stokes_j_label in enumerate(["T", "Q", "U"]):
                    cov_cmb[:, :, band_i_idx, stokes_i_idx, band_j_idx, stokes_j_idx] = (
                        cov_tqu_convolved[:, :, stokes_i_idx, stokes_j_idx]
                        / CMB_CALIBRATION_FACTORS[stokes_i_label][band_i_label]
                        / CMB_CALIBRATION_FACTORS[stokes_j_label][band_j_label]
                    )

    cov_cmb /= pixel_area_sr

    return cov_cmb


class PrecisionCalculator(ABC):
    """
    Abstract base class for precision calculators operating in Fourier space.

    Subclasses implement different models for estimating the per-source noise precision.
    """

    def __init__(self, config, map_shape):
        self.config = config
        self.n_bands = len(self.config.bands)
        self.map_shape = map_shape
        self._source_fields: Optional[np.ndarray] = None

    @abstractmethod
    def calculate_precision(
        self,
        maps_numpy: np.ndarray,
        idx_y: Optional[np.ndarray] = None,
        idx_x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Return Fourier-space precision tensors for the provided source maps.
        """

    def set_source_fields(self, source_fields: Optional[np.ndarray]) -> None:
        """Cache per-source field labels for calculators that operate per field."""
        if source_fields is None:
            self._source_fields = None
        else:
            self._source_fields = np.asarray(source_fields)

    @property
    def source_fields(self) -> Optional[np.ndarray]:
        """Return cached per-source field labels, if any."""
        return None if self._source_fields is None else np.asarray(self._source_fields)

    # ------------------------------------------------------------------ #
    # Shared helpers                                                     #
    # ------------------------------------------------------------------ #
    def _group_by_field(self, n_src: int) -> List[Tuple[Optional[Any], np.ndarray]]:
        """
        Return (field_label, indices) pairs covering all sources.

        If field metadata is missing or mismatched, treat the entire sample as one field.
        """
        labels = self.source_fields
        if labels is None or labels.shape[0] != n_src:
            return [(None, np.arange(n_src))]
        groups: List[Tuple[Optional[Any], np.ndarray]] = []
        for field in np.unique(labels):
            indices = np.where(labels == field)[0]
            if indices.size:
                groups.append((field, indices))
        return groups or [(None, np.arange(n_src))]

    def _apply_fieldwise(
        self,
        maps_numpy: np.ndarray,
        compute_fn,
    ) -> Tuple[np.ndarray, Dict[Any, Any]]:
        """
        Apply ``compute_fn`` to subsets of sources grouped by field.

        The callable must accept (field_maps, field_label, indices) and return
        (precision_per_source, debug_dict). precision_per_source must include a leading
        dimension equal to the number of indices for the current field.
        """
        n_src = maps_numpy.shape[0]
        groups = self._group_by_field(n_src)
        precision_out: Optional[np.ndarray] = None
        debug_out: Dict[Any, Any] = {}

        for field, indices in groups:
            field_maps = maps_numpy[indices]
            precision_field, debug_field = compute_fn(field_maps, field, indices)
            if precision_field.shape[0] != len(indices):
                raise ValueError("Field computation must return one precision grid per source.")
            if precision_out is None:
                precision_out = np.zeros((n_src,) + precision_field.shape[1:], dtype=precision_field.dtype)
            precision_out[indices] = precision_field
            if debug_field:
                key = field if field is not None else "all"
                debug_out[key] = debug_field

        return precision_out if precision_out is not None else np.zeros(0), debug_out

    def _truncate_fourier_numpy(
        self,
        array: Optional[np.ndarray],
        idx_y: Optional[np.ndarray],
        idx_x: Optional[np.ndarray],
        axis_y: int,
        axis_x: int,
    ) -> Optional[np.ndarray]:
        """Return the array truncated along the selected Fourier axes."""
        if array is None or idx_y is None or idx_x is None:
            return array
        truncated = np.take(array, idx_y, axis=axis_y)
        truncated = np.take(truncated, idx_x, axis=axis_x)
        return truncated

    def _diagonal_covariance_to_precision(self, covariance: np.ndarray, n_src: int) -> np.ndarray:
        """Return per-source precision grids for diagonal Fourier covariances."""
        covariance_array = np.asarray(covariance)
        if covariance_array.ndim == 4:
            covariance_array = np.broadcast_to(covariance_array, (n_src,) + covariance_array.shape)
        if covariance_array.ndim != 5:
            raise ValueError(f"Unexpected covariance shape {covariance_array.shape}; expected (..., ny, nx, n_bands, 3).")
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.reciprocal(covariance_array)
        precision = np.where(np.isfinite(precision), precision, 0.0)
        return precision.astype(self.config.dtype_np_real, copy=False)

    def _covariance_to_precision(self, covariance: np.ndarray) -> np.ndarray:
        """Invert covariance grids into precision tensors."""
        cov = np.asarray(covariance)
        if cov.ndim == 6:
            return self._invert_covariance_stack(cov)
        if cov.ndim == 7:
            return np.array([self._invert_covariance_stack(cov[i]) for i in range(cov.shape[0])])
        raise ValueError(f"Unexpected covariance shape {cov.shape}; expected (..., ny, nx, n_bands, 3, n_bands, 3).")

    def _invert_covariance_stack(self, covariance: np.ndarray) -> np.ndarray:
        """Invert a single covariance grid with axes (ny, nx, n_bands, 3, n_bands, 3)."""
        ny, nx, n_bands, n_stokes, _, _ = covariance.shape
        n_dim = n_bands * n_stokes
        assert covariance.dtype == self.config.dtype_np_real, f"Covariance must be {self.config.dtype_np_real}, not {covariance.dtype}"
        precision = np.zeros((ny, nx, n_bands, n_stokes, n_bands, n_stokes), dtype=self.config.dtype_np_real)
        for iy in range(ny):
            for ix in range(nx):
                cov = covariance[iy, ix].reshape(n_dim, n_dim)
                cov_spd = _project_to_spd(cov)
                inv = np.linalg.inv(cov_spd)
                inv = 0.5 * (inv + inv.T)  # probably don't need this
                precision[iy, ix] = inv.reshape(n_bands, n_stokes, n_bands, n_stokes)
        return precision


class ClusterfinderPSDCalculator(PrecisionCalculator):
    """
    Load pre-computed instrument noise PSD from clusterfinder analysis.

    This implementation loads a noise PSD from a FITS file that was pre-computed
    from clusterfinder instrument characterization data, then resamples it to match
    the analysis map resolution using mean-pooling.
    """

    def calculate_precision(
        self,
        maps_numpy: np.ndarray,
        idx_y: Optional[np.ndarray] = None,
        idx_x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Load the instrument PSD from disk, convert to precision, and broadcast to all sources."""
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

        precision = self._diagonal_covariance_to_precision(noise_psd_array, maps_numpy.shape[0])
        precision = self._truncate_fourier_numpy(precision, idx_y, idx_x, axis_y=1, axis_x=2)
        return precision, None

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
        # We know these have not been fftshift-ed so column 0 is the k_x=0 mode
        psd_resampled[:, 0] = 1e12

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


class KxAveragedCalculator(PrecisionCalculator):
    """
    Estimate diagonal Fourier covariances by averaging over k_y at fixed k_x.

    The estimator masks the central source region, computes per-source Fourier
    covariances, and combines them with a heuristic maximum to suppress noisy troughs
    before averaging across the ensemble of sources.
    """

    def calculate_precision(
        self,
        maps_numpy: np.ndarray,
        idx_y: Optional[np.ndarray] = None,
        idx_x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Estimate a diagonal precision model via kx-averaged covariance heuristics."""
        print("Calculating individual data-driven Fourier covariances for each source...")

        # Create noise mask with hole in center to avoid the source signal
        noise_mask = make_apod_mask_center_excised(
            self.map_shape,
            self.config.apodization_width_pix,
            self.config.noise_hole_radius_arcmin,
            self.config.reso_arcmin,
        )

        noise_covariances: List[np.ndarray] = []
        ny, nx = self.map_shape
        n_src = maps_numpy.shape[0]

        for i in range(n_src):
            print(f"  Processing source {i + 1}/{n_src}")

            # Create array for this source: (ky, kx, band, stokes)
            source_covariance = np.zeros((ny, nx, self.n_bands, 3), dtype=self.config.dtype_np_real)

            # Process each band and Stokes parameter
            for band_idx, band in enumerate(self.config.bands):
                for stokes_idx, stokes in enumerate(["T", "Q", "U"]):
                    # Extract data for this band and Stokes parameter
                    real_map = maps_numpy[i, :, :, band_idx, stokes_idx]

                    # Calculate individual Fourier covariance for this map
                    covariance_2d = self._calculate_individual_covariance(real_map, noise_mask)

                    # Apply scaling for polarization
                    if stokes in ["Q", "U"]:
                        covariance_2d *= 2.0  # Polarization has 2x the noise

                    source_covariance[:, :, band_idx, stokes_idx] = covariance_2d

            noise_covariances.append(source_covariance)

        # Average the covariances across all sources
        noise_covariances = np.array(noise_covariances)
        global_covariance = np.mean(noise_covariances, axis=0)

        print("Fourier covariance averaging complete.")
        precision = self._diagonal_covariance_to_precision(global_covariance, maps_numpy.shape[0])
        precision = self._truncate_fourier_numpy(precision, idx_y, idx_x, axis_y=1, axis_x=2)
        return precision, None

    def _calculate_individual_covariance(self, map_2d, noise_mask, sentinel_value=1e12):
        """
        Calculate a single-map Fourier covariance using k_y averaging.

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
            2D covariance array
        """
        # Apply noise mask to isolate empty regions
        masked_map = map_2d * noise_mask

        # Take FFT and calculate power spectrum
        fft_2d = np.fft.fft2(masked_map)
        covariance_2d = np.abs(fft_2d) ** 2

        # Average over k_y for each k_x
        ny, nx = covariance_2d.shape
        averaged_covariance = np.zeros_like(covariance_2d)

        for i in range(nx):
            # Average this column (constant k_x) over all k_y
            col_avg = np.mean(covariance_2d[:, i])
            averaged_covariance[:, i] = col_avg

        # Set k_x=0 modes to sentinel value to avoid division by zero
        averaged_covariance[:, 0] = sentinel_value

        # Take element-wise maximum to avoid scattered low values (heuristic)
        covariance = np.maximum(averaged_covariance, covariance_2d)

        return covariance


class MeanAmplitudeSpectrumCalculator(PrecisionCalculator):
    """
    Estimate diagonal Fourier covariances by averaging amplitude spectra across sources.

    Each masked source map is transformed to the Fourier domain, converted to an amplitude
    spectrum, averaged over the ensemble, and squared to recover a covariance model.
    """

    def calculate_precision(
        self,
        maps_numpy: np.ndarray,
        idx_y: Optional[np.ndarray] = None,
        idx_x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Estimate diagonal precision via ensemble-averaged amplitude spectra."""
        print("Calculating ensemble-averaged amplitude spectra...")

        # Create noise mask with hole in center to avoid the source signal
        noise_mask = make_apod_mask_center_excised(
            self.map_shape,
            self.config.apodization_width_pix,
            self.config.noise_hole_radius_arcmin,
            self.config.reso_arcmin,
        )

        # Collect amplitude spectra from all sources
        ny, nx = self.map_shape
        n_src = maps_numpy.shape[0]
        all_amplitudes = np.zeros((n_src, ny, nx, self.n_bands, 3), dtype=self.config.dtype_np_real)

        for i in range(n_src):
            print(f"  Processing source {i + 1}/{n_src}")

            for band_idx, band in enumerate(self.config.bands):
                for stokes_idx, stokes in enumerate(["T", "Q", "U"]):
                    real_map = maps_numpy[i, :, :, band_idx, stokes_idx]

                    # Apply center-excised mask
                    masked_map = real_map * noise_mask

                    # Calculate covariance and convert to amplitude spectrum
                    fft_2d = np.fft.fft2(masked_map)
                    covariance_2d = np.abs(fft_2d) ** 2

                    # Convert covariance to amplitude spectrum
                    amplitude_spectrum_2d = np.sqrt(covariance_2d)
                    all_amplitudes[i, :, :, band_idx, stokes_idx] = amplitude_spectrum_2d

        # Average amplitude spectra across all sources and convert back to covariance
        print("  Averaging amplitude spectra across sources...")
        mean_amplitude_spectrum = np.mean(all_amplitudes, axis=0)  # Average over sources
        mean_covariance = mean_amplitude_spectrum**2

        print("Ensemble amplitude spectrum averaging complete.")
        precision = self._diagonal_covariance_to_precision(mean_covariance, maps_numpy.shape[0])
        precision = self._truncate_fourier_numpy(precision, idx_y, idx_x, axis_y=1, axis_x=2)
        return precision, None


class CmbPcaCalculator(PrecisionCalculator):
    """Subtract CMB covariance and PCA-regularize residual diagonals per field."""

    def calculate_precision(
        self,
        maps_numpy: np.ndarray,
        idx_y: Optional[np.ndarray] = None,
        idx_x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        maps_numpy = np.asarray(maps_numpy, dtype=self.config.dtype_np_real)
        n_src, ny_full, nx_full, n_bands, n_stokes = maps_numpy.shape
        print(f"maps_numpy shape: {maps_numpy.shape}")
        print(f"Computing covariance for {n_src} sources across {n_bands} bands and {n_stokes} Stokes parameters.")

        noise_mask = make_apod_mask_center_excised(
            (ny_full, nx_full),
            self.config.apodization_width_pix,
            self.config.noise_hole_radius_arcmin,
            self.config.reso_arcmin,
        ).astype(self.config.dtype_np_real)

        if idx_y is None or idx_x is None:
            idx_y, idx_x = _ensure_fft_cut_indices((ny_full, nx_full), self.config)
        idx_y_eff, idx_x_eff = idx_y, idx_x
        sliced_to_indices = idx_y_eff is not None and idx_x_eff is not None
        ky_full, kx_full, ky_grid_full, kx_grid_full = compute_fourier_frequency_axes((ny_full, nx_full), self.config.reso_arcmin)

        masked_maps = maps_numpy * noise_mask[None, :, :, None, None]
        fft_maps = np.fft.fft2(masked_maps, axes=(1, 2))

        if sliced_to_indices:
            fft_maps = np.take(fft_maps, idx_y_eff, axis=1)
            fft_maps = np.take(fft_maps, idx_x_eff, axis=2)
            ky_grid = ky_grid_full[np.ix_(idx_y_eff, idx_x_eff)]
            kx_grid = kx_grid_full[np.ix_(idx_y_eff, idx_x_eff)]
        else:
            ky_grid, kx_grid = ky_grid_full, kx_grid_full

        ell_x_grid = 360.0 * kx_grid
        ell_y_grid = 360.0 * ky_grid
        ell_radial = np.sqrt(ell_x_grid**2 + ell_y_grid**2)

        covariance = np.einsum("nyxbs,nyxct->nyxbsct", fft_maps, np.conj(fft_maps), optimize=True).real.astype(self.config.dtype_np_real)

        print(f"Largest ell_x value in covariance: {float(np.max(ell_x_grid)):.2f}")
        _print_rms_summary("Average noise (all ell modes)", covariance, self.config)

        ny, nx = ell_x_grid.shape
        print("Calculating expected CMB covariance...")
        cov_cmb = _compute_cmb_covariance(self.config, ny, nx)
        _print_rms_summary("Average CMB noise (all ell modes)", cov_cmb[None, ...], self.config)

        cov_no_cmb = covariance - cov_cmb[None, :, :, :, :, :, :]
        _print_rms_summary("Average noise after CMB subtraction (all ell modes)", cov_no_cmb, self.config)

        diag_original = self._extract_diagonals(cov_no_cmb)
        print("Applying PCA to regularize band-stokes diagonal noise covariances...")
        diag_regularized = self._regularize_diagonals(diag_original, ell_x_grid, ell_y_grid)
        print("PCA regularization complete.")

        print("Estimating white-noise floors from residual (non-CMB) covariance...")
        floors = self._compute_white_noise_floors(diag_original, ell_x_grid, ell_y_grid)
        diag_bounded = self._apply_white_noise_floors(diag_regularized, floors)

        residual = np.zeros_like(cov_no_cmb)
        for band in range(n_bands):
            for stokes in range(3):
                residual[:, :, :, band, stokes, band, stokes] = diag_bounded[:, :, :, band, stokes]

        covariance_model = residual + cov_cmb[None, :, :, :, :, :, :]
        _print_rms_summary("Average noise after PCA regularization (all ell modes)", residual, self.config)
        _print_rms_summary("Average noise after PCA regularization and adding CMB (all ell modes)", covariance_model, self.config)
        print("Covariance matrix calculation complete.")

        precision = self._covariance_to_precision(covariance_model)
        if not sliced_to_indices:
            precision = self._truncate_fourier_numpy(precision, idx_y_eff, idx_x_eff, axis_y=1, axis_x=2)

        debug = None
        if self.config.debug:
            debug = {
                "covariance_raw": covariance,
                "covariance_cmb": cov_cmb,
                "covariance_noncmb": residual,
                "covariance_total": covariance_model,
                "white_noise_floors": floors,
                "precision": precision,
                "k_indices_y": None if idx_y_eff is None else np.asarray(idx_y_eff),
                "k_indices_x": None if idx_x_eff is None else np.asarray(idx_x_eff),
                "ell_x_grid": ell_x_grid,
                "ell_y_grid": ell_y_grid,
                "ell_radial": ell_radial,
            }
        return precision, debug

    def _extract_diagonals(self, cov_no_cmb: np.ndarray) -> np.ndarray:
        """Return residual covariance diagonals with shape (n_src, ny, nx, n_bands, 3)."""
        n_src, ny, nx, n_bands, _, _, _ = cov_no_cmb.shape
        diag = np.zeros((n_src, ny, nx, n_bands, 3), dtype=cov_no_cmb.dtype)
        for band in range(n_bands):
            for stokes in range(3):
                diag[:, :, :, band, stokes] = cov_no_cmb[:, :, :, band, stokes, band, stokes]
        return diag

    def _regularize_diagonals(self, diagonals: np.ndarray, ell_x_grid: np.ndarray, ell_y_grid: np.ndarray) -> np.ndarray:
        """PCA-regularize residual diagonals per band and Stokes, grouped by field."""
        n_src, ny, nx, n_bands, n_stokes = diagonals.shape
        field_labels = self._resolve_field_labels(n_src)
        # convert from complex to real
        diagonals = np.real(diagonals).astype(np.float64)  # real part of input
        regularized = np.zeros_like(diagonals, dtype=np.float64)  # output

        for field in np.unique(field_labels):
            indices = np.where(field_labels == field)[0]
            if indices.size == 0:
                print(f"Warning, no sources found for field {field}!")
            print(f"  PCA regularization for field {field} with {indices.size} sources.")
            for band in range(n_bands):
                for stokes in range(n_stokes):
                    samples = diagonals[indices, :, :, band, stokes].reshape(indices.size, -1)
                    mean = samples.mean(axis=0, dtype=np.float64)
                    centered = samples - mean
                    n_components = self.config.n_pca_components
                    if n_components >= indices.size:
                        print(
                            f"Warning: n_pca_components ({n_components}) is >= number of sources ({indices.size}) for field {field}. "
                            f"Reducing to {indices.size - 1} to avoid overfitting."
                        )
                        n_components = indices.size - 1
                    if n_components > 0:
                        print(f"Applying PCA order {n_components} to field {field} band {band} stokes {stokes}.")
                        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
                        transformed = pca.fit_transform(centered)
                        reconstructed = pca.inverse_transform(transformed)
                    else:  # allow for no PCA
                        print(f"No PCA applied to field {field} band {band} stokes {stokes}.")
                        reconstructed = np.zeros_like(centered)

                    samples_reconstructed = reconstructed + mean

                    regularized[indices, :, :, band, stokes] = samples_reconstructed.reshape(indices.size, ny, nx)
        return regularized

    def _resolve_field_labels(self, n_src: int) -> np.ndarray:
        """Return per-source field labels aligned with the provided maps."""
        if self.source_fields is None:
            return np.zeros(n_src, dtype=int)
        labels = np.asarray(self.source_fields, dtype=object)
        if labels.shape[0] != n_src:
            print(f"Length of source_fields ({labels.shape[0]}) does not match number of sources ({n_src}); treating all as one field.")
            return np.zeros(n_src, dtype=int)
        return labels

    def _compute_white_noise_floors(
        self,
        diagonals: np.ndarray,
        ell_x_grid: np.ndarray,
        ell_y_grid: np.ndarray,
    ) -> np.ndarray:
        """Return per-band, per-stokes floor values based on high-ell statistics."""
        n_src, ny, nx, n_bands, n_stokes = diagonals.shape
        region_mask = (ell_y_grid > 3000.0) & (ell_x_grid > 3000.0) & (ell_x_grid < 8000.0)
        if not np.any(region_mask):
            print("cmb_pca white-noise region mask is empty; falling back to full grid for floor estimation.")
            region_mask = np.ones_like(ell_x_grid, dtype=bool)
        region_flat = region_mask.reshape(ny * nx)
        diag_flat = diagonals.reshape(n_src, ny * nx, n_bands, n_stokes)

        floors = np.zeros((n_bands, n_stokes), dtype=diagonals.dtype)
        for band in range(n_bands):
            for stokes in range(n_stokes):
                values = diag_flat[:, region_flat, band, stokes]

                positive_means = []
                for src in range(n_src):
                    slice_vals = values[src]
                    if slice_vals.size == 0:
                        continue
                    finite_vals = slice_vals[np.isfinite(slice_vals) & (slice_vals > 0.0)]
                    if finite_vals.size == 0:
                        continue
                    positive_means.append(float(finite_vals.mean()))

                white_level = None
                if positive_means:
                    white_level = np.percentile(positive_means, 20.0)
                else:
                    fallback_vals = diagonals[:, :, :, band, stokes].reshape(-1)
                    fallback_vals = fallback_vals[np.isfinite(fallback_vals) & (fallback_vals > 0.0)]
                    if fallback_vals.size:
                        white_level = np.percentile(fallback_vals, 20.0)

                if white_level is None or white_level <= 0.0:
                    fallback_abs = np.abs(diagonals[:, :, :, band, stokes]).reshape(-1)
                    fallback_abs = fallback_abs[np.isfinite(fallback_abs)]
                    if fallback_abs.size:
                        white_level = np.percentile(fallback_abs, 50.0) * 0.1
                    else:
                        white_level = np.finfo(diagonals.dtype).eps

                eps_floor = np.finfo(diagonals.dtype).eps
                floors[band, stokes] = max(white_level * 0.8, eps_floor)
        return floors

    def _apply_white_noise_floors(self, diagonals: np.ndarray, floors: np.ndarray) -> np.ndarray:
        """Bound residual diagonals from below using the provided white-noise floor estimates."""
        bounded = np.array(diagonals, copy=True)
        n_src, ny, nx, n_bands, n_stokes = bounded.shape
        for band in range(n_bands):
            for stokes in range(n_stokes):
                floor = float(floors[band, stokes])
                label = f"{'TQU'[stokes]}{_band_suffix(self.config.bands[band])}"
                if floor <= 0:
                    raise ValueError(f"White noise floor for {label} is non-positive: {floor}")
                floor_rms = np.sqrt(floor) / self.config.map_size_pix * 1000.0 * self.config.reso_arcmin
                print(f"White noise floor {label}: {floor_rms:.2f} µK-arcmin")
                below_floor_mask = bounded[:, :, :, band, stokes] < floor
                if np.any(below_floor_mask):
                    below_floor_fraction = np.sum(below_floor_mask) / np.prod(below_floor_mask.shape)
                    print(
                        f"Some covariance values are below the white noise floor {label}. "
                        f"{below_floor_fraction:.2%} of the values are below the floor. Clipping to the floor and continuing..."
                    )
                    bounded[:, :, :, band, stokes][below_floor_mask] = floor
        return bounded


class PcaCalculator(PrecisionCalculator):
    """
    Perform separate PCA analyses for each Stokes parameter in log-covariance space.

    The estimator models T, Q, and U covariances independently, allowing for distinct
    noise structures in each Stokes component.
    """

    def calculate_precision(
        self,
        maps_numpy: np.ndarray,
        idx_y: Optional[np.ndarray] = None,
        idx_x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Calculate Fourier covariances using separate log-space PCA models for T, Q, and U."""
        print(f"Calculating Fourier covariances with separate PCA models (components per Stokes: {self.config.n_pca_components})...")
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
        per_source_covariance = np.zeros_like(maps_numpy, dtype=self.config.dtype_np_real)

        # Process each Stokes parameter separately (T, Q, U)
        stokes_names = ["T", "Q", "U"]
        for stokes_idx, stokes_name in enumerate(stokes_names):
            print(f"Collating {stokes_name} Fourier covariances...")
            all_covariances_flat = []
            for i in range(n_src):
                for band_idx in range(n_bands):
                    real_map = maps_numpy[i, :, :, band_idx, stokes_idx]
                    masked_map = real_map * noise_mask
                    fft_2d = np.fft.fft2(masked_map)
                    covariance_2d = np.abs(fft_2d) ** 2
                    covariance_2d *= (ny * nx) / effective_area  # correct for mask
                    all_covariances_flat.append(covariance_2d.flatten())

            X_linear = np.array(all_covariances_flat)
            X_log = np.log(X_linear)
            print(f"Performing PCA on {stokes_name} log-transformed covariances...")
            mean_log_covariance = np.mean(X_log, axis=0)
            X_log_centered = X_log - mean_log_covariance
            pca = PCA(n_components=self.config.n_pca_components, svd_solver="randomized", random_state=42)
            pca.fit(X_log_centered)
            print(f"{stokes_name} PCA explained variance ratio: {pca.explained_variance_ratio_}")

            print(f"Reconstructing denoised {stokes_name} covariances...")
            coeffs = pca.transform(X_log_centered)
            X_reconstructed_log_centered = pca.inverse_transform(coeffs)
            X_reconstructed_log = X_reconstructed_log_centered + mean_log_covariance
            X_reconstructed_linear = np.exp(X_reconstructed_log)

            # Fill this Stokes component in output array
            map_idx = 0
            for i in range(n_src):
                for band_idx in range(n_bands):
                    per_source_covariance[i, :, :, band_idx, stokes_idx] = X_reconstructed_linear[map_idx].reshape(map_shape)
                    map_idx += 1

        print("Separate T, Q, and U PCA-based covariance calculation complete.")
        precision = self._diagonal_covariance_to_precision(per_source_covariance, maps_numpy.shape[0])
        precision = self._truncate_fourier_numpy(precision, idx_y, idx_x, axis_y=1, axis_x=2)
        return precision, None


class WhiteNoiseCalculator(PrecisionCalculator):
    """
    Produce a simple white-noise covariance with constant Fourier amplitude.

    Useful for smoke tests and baseline comparisons.
    """

    def calculate_precision(
        self,
        maps_numpy: np.ndarray,
        idx_y: Optional[np.ndarray] = None,
        idx_x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        print("Generating simple white-noise covariance...")

        ny, nx = self.map_shape

        # Create white noise covariance array with all ones
        noise_covariance_array = np.ones((ny, nx, self.n_bands, 3), dtype=self.config.dtype_np_real)

        # Apply scaling for polarization
        noise_covariance_array[:, :, :, 1] *= 2.0  # Q polarization has 2x the noise
        noise_covariance_array[:, :, :, 2] *= 2.0  # U polarization has 2x the noise

        print("White-noise covariance ready.")
        precision = self._diagonal_covariance_to_precision(noise_covariance_array, maps_numpy.shape[0])
        precision = self._truncate_fourier_numpy(precision, idx_y, idx_x, axis_y=1, axis_x=2)
        return precision, None


# Factory function to create appropriate precision calculator
def create_precision_calculator(config, map_shape):
    """
    Factory function to create the appropriate precision calculator based on configuration.

    Parameters:
    -----------
    config : BeamFittingConfig
        Configuration object containing band and other parameters
    map_shape : tuple
        Shape of the maps (ny, nx)

    Returns:
    --------
    PrecisionCalculator
        Appropriate precision calculator instance
    """
    if config.covariance_method == "clusterfinder_psd":
        return ClusterfinderPSDCalculator(config, map_shape)
    elif config.covariance_method == "kx_averaged":
        return KxAveragedCalculator(config, map_shape)
    elif config.covariance_method == "mean_amplitude_spectrum":
        return MeanAmplitudeSpectrumCalculator(config, map_shape)
    elif config.covariance_method == "pca":
        return PcaCalculator(config, map_shape)
    elif config.covariance_method == "cmb_pca":
        return CmbPcaCalculator(config, map_shape)
    elif config.covariance_method == "white_noise":
        return WhiteNoiseCalculator(config, map_shape)
    else:
        raise ValueError(f"Unknown covariance_method: {config.covariance_method}")
