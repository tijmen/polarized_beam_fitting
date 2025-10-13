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
and [multiband_covariance] is only diagonal in Fourier space. # TODO update this docstring
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import camb
import numpy as np
from astropy.io import fits
from sklearn.decomposition import PCA
from spt3g import core

from .utils import (
    compute_fourier_frequency_axes,
    compute_rectangular_ell_cut_indices,
    make_apod_mask_center_excised,
)

PARAMETRIC_PRECISION_SUPERSAMPLING = 8
PARAMETRIC_PRECISION_HIGH_KY_THRESHOLD = 3000.0
PARAMETRIC_PRECISION_WHITE_NOISE_FLOOR_ELL = 8000.0
PARAMETRIC_PRECISION_RADIAL_MAX_ELL = 3000.0
PARAMETRIC_PRECISION_RADIAL_PIVOT = 1000.0
PARAMETRIC_PRECISION_RADIAL_EXPONENT = 1.0
PARAMETRIC_PRECISION_RADIAL_DC_FAKE = 50.0
PARAMETRIC_PRECISION_EIGEN_EPS = 1e-5
PARAMETRIC_PRECISION_DESCRIPTION = (
    "Precision matrix from parametric modelling: CMB (fixed calibration) + ell_x-dependent + radial uncorrelated model. "
    "Uses unshifted FFT convention (DC located at [0,0]) and discrete Fourier normalization."
)


def _ensure_fft_cut_indices(map_shape: Tuple[int, int], config) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return Fourier indices consistent with the configured ellmax cutoff."""
    idx_y, idx_x = compute_rectangular_ell_cut_indices(map_shape, config.reso_arcmin, getattr(config, "ellmax", None))
    if idx_y is None or idx_x is None:
        return None, None
    return idx_y, idx_x


def _smooth_highpass_1d(ell_vals: np.ndarray, ell0: float = 360.0, ell1: float = 420.0) -> np.ndarray:
    """Raised-cosine high-pass filter used to suppress large-scale CMB power."""
    H = np.ones_like(ell_vals, dtype=float)
    H[ell_vals <= ell0] = 0.0
    transition = (ell_vals > ell0) & (ell_vals < ell1)
    if np.any(transition):
        frac = (ell_vals[transition] - ell0) / (ell1 - ell0)
        H[transition] = 0.5 * (1.0 - np.cos(np.pi * frac))
    return H


def _default_cmb_calibration(config, n_bands: int) -> np.ndarray:
    """Return per-stokes calibration factors to convert between bandpass calibrations."""
    cal = getattr(config, "cmb_calibration_factors", None)
    if cal is None:
        cal = np.array(
            [
                [1.07, 1.02, 1.01],
                [1.05, 1.06, 1.17],
                [1.05, 1.06, 1.17],
            ],
            dtype=config.dtype_np_real,
        )
    cal = np.asarray(cal, dtype=config.dtype_np_real)
    if cal.shape[0] != 3:
        raise ValueError(f"Expected cmb_calibration_factors with 3 stokes rows; received shape {cal.shape}.")
    if cal.shape[1] < n_bands:
        raise ValueError(
            f"Expected cmb_calibration_factors with at least {n_bands} columns, received {cal.shape}. "
            "Update the configuration to match the active band list."
        )
    if cal.shape[1] > n_bands:
        cal = cal[:, :n_bands]
    return cal


def _project_to_spd(matrix: np.ndarray, eps: float = PARAMETRIC_PRECISION_EIGEN_EPS) -> np.ndarray:
    """Project a covariance matrix to the nearest symmetric positive-definite matrix."""
    symm = 0.5 * (matrix + matrix.T)
    diag = np.diag(symm)
    positive_diag = diag[diag > 0]
    scale = np.median(positive_diag) if positive_diag.size else 1.0
    floor = eps * scale
    eigvals, eigvecs = np.linalg.eigh(symm)
    eigvals = np.maximum(eigvals, floor)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _compute_covariance_periodogram(
    maps_numpy: np.ndarray,
    config,
    noise_mask: np.ndarray,
    idx_y: Optional[np.ndarray],
    idx_x: Optional[np.ndarray],
    Omega_pix: float,
    Omega_eff: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the per-source covariance periodogram after masking and FFT."""
    masked_maps = maps_numpy * noise_mask[None, :, :, None, None]
    fft = np.fft.fft2(masked_maps, axes=(1, 2)) * Omega_pix

    ny_full, nx_full = maps_numpy.shape[1:3]
    ky_full, kx_full, ky_grid_full, kx_grid_full = compute_fourier_frequency_axes((ny_full, nx_full), config.reso_arcmin)

    if idx_y is not None and idx_x is not None:
        fft = np.take(fft, idx_y, axis=1)
        fft = np.take(fft, idx_x, axis=2)
        ky = ky_full[idx_y]
        kx = kx_full[idx_x]
        ky_grid = ky_grid_full[np.ix_(idx_y, idx_x)]
        kx_grid = kx_grid_full[np.ix_(idx_y, idx_x)]
    else:
        ky, kx, ky_grid, kx_grid = ky_full, kx_full, ky_grid_full, kx_grid_full

    covariance = np.einsum("nyxbs,nyxct->nyxbsct", fft, np.conj(fft), optimize=True).real.astype(config.dtype_np_real) / Omega_eff
    ell_x = 360.0 * kx
    ell_y = 360.0 * ky
    ell_x_grid = 360.0 * kx_grid
    ell_y_grid = 360.0 * ky_grid
    ell_radial = np.sqrt(ell_x_grid**2 + ell_y_grid**2)
    return covariance, ell_x, ell_y, ell_x_grid, ell_y_grid, ell_radial


def _compute_cmb_covariance(
    config,
    ny: int,
    nx: int,
    ell_x_grid: np.ndarray,
    ell_y_grid: np.ndarray,
    ell_radial: np.ndarray,
    n_bands: int,
) -> np.ndarray:
    """Build the CMB covariance grid in T/Q/U and expand to band × band × stokes × stokes."""
    logging.info("Computing CAMB spectra for parametric precision model...")
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.1200, mnu=0.06, omk=0, tau=0.0544)
    pars.InitPower.set_params(As=2.1e-9, ns=0.9649)
    ell_max_cmb = 5000
    pars.set_for_lmax(ell_max_cmb, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)
    ell_camb = np.arange(powers["total"].shape[0])

    cl_tt_raw = powers["total"][:, 0]
    cl_ee_raw = powers["total"][:, 1]
    cl_bb_raw = powers["total"][:, 2]
    cl_te_raw = powers["total"][:, 3]

    hp_filter = _smooth_highpass_1d(ell_camb, 360.0, 420.0)
    cl_tt = cl_tt_raw * hp_filter
    cl_ee = cl_ee_raw * hp_filter
    cl_bb = cl_bb_raw * hp_filter
    cl_te = cl_te_raw * hp_filter

    ny_highres = ny * PARAMETRIC_PRECISION_SUPERSAMPLING
    nx_highres = nx * PARAMETRIC_PRECISION_SUPERSAMPLING

    dell_x = ell_x_grid[0, 1] - ell_x_grid[0, 0] if nx > 1 else 360.0 / (nx * config.reso_arcmin / 60.0)
    dell_y = ell_y_grid[1, 0] - ell_y_grid[0, 0] if ny > 1 else 360.0 / (ny * config.reso_arcmin / 60.0)

    ell_x_highres = np.fft.fftfreq(nx_highres, d=1.0 / (nx_highres * dell_x / PARAMETRIC_PRECISION_SUPERSAMPLING))
    ell_y_highres = np.fft.fftfreq(ny_highres, d=1.0 / (ny_highres * dell_y / PARAMETRIC_PRECISION_SUPERSAMPLING))
    ell_x_grid_highres, ell_y_grid_highres = np.meshgrid(ell_x_highres, ell_y_highres, indexing="xy")
    ell_radial_highres = np.sqrt(ell_x_grid_highres**2 + ell_y_grid_highres**2)

    def interp(cls):
        return np.interp(ell_radial_highres.ravel(), ell_camb, cls, left=0.0, right=0.0).reshape(ell_radial_highres.shape)

    C_TT = interp(cl_tt)
    C_EE = interp(cl_ee)
    C_BB = interp(cl_bb)
    C_TE = interp(cl_te)

    phi = np.arctan2(ell_y_grid_highres, ell_x_grid_highres)
    c2phi = np.cos(2.0 * phi)
    s2phi = np.sin(2.0 * phi)

    cov_tqu_highres = np.zeros((ny_highres, nx_highres, 3, 3), dtype=config.dtype_np_real)
    cov_tqu_highres[..., 0, 0] = C_TT
    cov_tqu_highres[..., 0, 1] = cov_tqu_highres[..., 1, 0] = C_TE * c2phi
    cov_tqu_highres[..., 0, 2] = cov_tqu_highres[..., 2, 0] = C_TE * s2phi
    cov_tqu_highres[..., 1, 1] = C_EE * c2phi**2 + C_BB * s2phi**2
    cov_tqu_highres[..., 2, 2] = C_EE * s2phi**2 + C_BB * c2phi**2
    cov_tqu_highres[..., 1, 2] = cov_tqu_highres[..., 2, 1] = (C_EE - C_BB) * s2phi * c2phi

    cov_shifted = np.fft.fftshift(cov_tqu_highres, axes=(0, 1))
    cov_shifted = cov_shifted.reshape(ny, PARAMETRIC_PRECISION_SUPERSAMPLING, nx, PARAMETRIC_PRECISION_SUPERSAMPLING, 3, 3).mean(
        axis=(1, 3)
    )
    cov_tqu = np.fft.ifftshift(cov_shifted, axes=(0, 1))

    cal = _default_cmb_calibration(config, n_bands)
    cov_cmb = np.zeros((ny, nx, n_bands, 3, n_bands, 3), dtype=config.dtype_np_real)
    cal_inv = 1.0 / cal
    for s1 in range(3):
        for s2 in range(3):
            factor = np.outer(cal_inv[s1], cal_inv[s2])  # (band, band)
            cov_cmb[:, :, :, s1, :, s2] = cov_tqu[:, :, s1, s2][:, :, None, None] * factor[None, None, :, :]

    return cov_cmb


def _vandermonde_matrix(ell_x_vals: np.ndarray, pivot: float = 1000.0, ell_min: float = 100.0) -> np.ndarray:
    """Return the fixed design matrix used for ell_x polynomial fitting."""
    ell_x_vals = np.asarray(ell_x_vals, dtype=float)
    a = ell_x_vals / pivot
    design = np.zeros((len(ell_x_vals), 6), dtype=float)

    is_dc = ell_x_vals == 0.0
    is_high = np.abs(ell_x_vals) >= ell_min
    is_low = (~is_dc) & (~is_high)

    aa_high = a[is_high]
    design[is_high, 0] = 1.0 / (aa_high**4)
    design[is_high, 1] = 1.0 / np.abs(aa_high)
    design[is_high, 2] = 1.0
    design[is_high, 3] = aa_high**2
    design[is_high, 4] = aa_high**4

    aa_low = a[is_low]
    design[is_low, 2] = 1.0
    design[is_low, 3] = aa_low**2
    design[is_low, 4] = aa_low**4

    design[is_dc, 5] = 1.0
    return design


def _fit_ell_x_model(
    cov_no_cmb: np.ndarray,
    cov_input_for_floor: np.ndarray,
    ell_x: np.ndarray,
    high_ky_mask: np.ndarray,
) -> np.ndarray:
    """Fit the ell_x-dependent uncorrelated noise model for each source/band/stokes."""
    n_src, _, nx, n_bands, n_stokes, _, _ = cov_no_cmb.shape
    design = _vandermonde_matrix(ell_x)
    cov_ellx = np.zeros((n_src, nx, n_bands, n_stokes, n_bands, n_stokes), dtype=cov_no_cmb.dtype)

    if not np.any(high_ky_mask):
        logging.warning("High-k_y mask for ell_x fitting is empty; falling back to full grid.")
        averaged = np.mean(cov_no_cmb, axis=1)
    else:
        averaged = np.mean(cov_no_cmb[:, high_ky_mask, :, :, :, :, :], axis=1)

    for src in range(n_src):
        for band in range(n_bands):
            for stokes in range(n_stokes):
                y = averaged[src, :, band, stokes, band, stokes]
                theta, *_ = np.linalg.lstsq(design, y, rcond=None)
                model = design @ theta
                model = np.maximum(model, 0.0)

                white_mask = np.abs(ell_x) >= PARAMETRIC_PRECISION_WHITE_NOISE_FLOOR_ELL
                if np.any(white_mask):
                    reference = cov_input_for_floor[src, :, :, band, stokes, band, stokes][:, white_mask]
                    if reference.size:
                        floor = 0.7 * np.median(reference)
                        model = np.where(white_mask, np.maximum(model, floor), model)
                cov_ellx[src, :, band, stokes, band, stokes] = model
    return cov_ellx


def _fit_radial_model(
    cov_no_cmb_ellx_sub: np.ndarray,
    ell_radial: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the radial red-noise model and reconstruct on the full grid."""
    n_src, ny, nx, n_bands, n_stokes, _, _ = cov_no_cmb_ellx_sub.shape
    ell_vals = np.where(ell_radial == 0.0, PARAMETRIC_PRECISION_RADIAL_DC_FAKE, ell_radial)
    mask = ell_vals <= PARAMETRIC_PRECISION_RADIAL_MAX_ELL
    basis = (PARAMETRIC_PRECISION_RADIAL_PIVOT / ell_vals[mask]) ** PARAMETRIC_PRECISION_RADIAL_EXPONENT
    denom = np.dot(basis, basis)
    if denom <= 0:
        logging.warning("Radial basis denominator non-positive; skipping radial fit.")
        amplitudes = np.zeros((n_src, n_bands, n_stokes), dtype=cov_no_cmb_ellx_sub.dtype)
        model = np.zeros_like(cov_no_cmb_ellx_sub)
        return amplitudes, model

    amplitudes = np.zeros((n_src, n_bands, n_stokes), dtype=cov_no_cmb_ellx_sub.dtype)
    radial_model = np.zeros_like(cov_no_cmb_ellx_sub)
    basis_full = (
        PARAMETRIC_PRECISION_RADIAL_PIVOT / np.maximum(ell_vals, PARAMETRIC_PRECISION_RADIAL_DC_FAKE)
    ) ** PARAMETRIC_PRECISION_RADIAL_EXPONENT

    for src in range(n_src):
        for band in range(n_bands):
            for stokes in range(n_stokes):
                data = cov_no_cmb_ellx_sub[src, :, :, band, stokes, band, stokes][mask]
                amplitude = np.dot(basis, data) / denom
                if amplitude < 0:
                    amplitude = 0.0
                amplitudes[src, band, stokes] = amplitude
                radial_model[src, :, :, band, stokes, band, stokes] = amplitude * basis_full
    return amplitudes, radial_model


def compute_parametric_precision(config, raw_maps: np.ndarray) -> Dict[str, Any]:
    """
    Compute the parametric precision matrix using the production pipeline formerly implemented in precision.ipynb.

    Parameters
    ----------
    config : BeamFittingConfig
        Analysis configuration.
    raw_maps : np.ndarray
        Array with shape (n_src, ny, nx, n_bands, 3) containing the per-source raw maps in G3 units.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the precision matrix and supporting metadata useful for diagnostics.
    """
    logging.info("Starting parametric precision computation...")

    maps_numpy = np.asarray(raw_maps, dtype=config.dtype_np_real) / core.G3Units.uK
    n_src, ny_full, nx_full, n_bands, n_stokes = maps_numpy.shape
    if n_stokes != 3:
        raise ValueError(f"Parametric precision pipeline expects 3 Stokes parameters; received {n_stokes}.")

    noise_mask = make_apod_mask_center_excised(
        (ny_full, nx_full),
        config.apodization_width_pix,
        config.noise_hole_radius_arcmin,
        config.reso_arcmin,
    )

    dtheta_rad = config.reso_arcmin * (np.pi / (180.0 * 60.0))
    Omega_pix = dtheta_rad**2
    Omega_eff = Omega_pix * float(np.sum(noise_mask**2))

    idx_y, idx_x = _ensure_fft_cut_indices((ny_full, nx_full), config)
    covariance, ell_x, ell_y, ell_x_grid, ell_y_grid, ell_radial = _compute_covariance_periodogram(
        maps_numpy,
        config,
        noise_mask.astype(config.dtype_np_real),
        idx_y,
        idx_x,
        Omega_pix,
        Omega_eff,
    )
    ny, nx = ell_x_grid.shape

    cov_cmb = _compute_cmb_covariance(config, ny, nx, ell_x_grid, ell_y_grid, ell_radial, n_bands)
    cov_no_cmb = covariance - cov_cmb[None, :, :, :, :, :, :]

    high_ky_mask = ell_y > PARAMETRIC_PRECISION_HIGH_KY_THRESHOLD
    cov_ell_x_model = _fit_ell_x_model(cov_no_cmb, covariance, ell_x, high_ky_mask)
    cov_no_cmb_ellx_sub = cov_no_cmb - cov_ell_x_model[:, None, :, :, :, :, :]

    _, cov_radial_model = _fit_radial_model(cov_no_cmb_ellx_sub, ell_radial)

    cov_model_full = cov_cmb[None, :, :, :, :, :, :] + cov_ell_x_model[:, None, :, :, :, :, :] + cov_radial_model

    precision = np.zeros_like(cov_model_full, dtype=config.dtype_np_complex)
    n_dim = n_bands * n_stokes
    logging.info("Inverting covariance model to precision matrices...")
    for src in range(n_src):
        for iy in range(ny):
            for ix in range(nx):
                cov_9x9 = cov_model_full[src, iy, ix].reshape(n_dim, n_dim)
                cov_spd = _project_to_spd(cov_9x9, PARAMETRIC_PRECISION_EIGEN_EPS)
                try:
                    prec = np.linalg.inv(cov_spd)
                except np.linalg.LinAlgError:
                    logging.warning(f"Singular covariance matrix at src {src}, iy {iy}, ix {ix}. Using pseudo-inverse.")
                    prec = np.linalg.pinv(cov_spd)
                prec = 0.5 * (prec + prec.T)
                precision[src, iy, ix] = prec.reshape(n_bands, 3, n_bands, 3)

    precision *= Omega_pix
    precision /= core.G3Units.uK**2

    logging.info("Parametric precision computation complete.")
    return {
        "precision": precision.astype(config.dtype_np_complex),
        "covariance_model": cov_model_full.astype(config.dtype_np_real),
        "covariance_components": {
            "cmb": cov_cmb.astype(config.dtype_np_real),
            "ell_x_model": cov_ell_x_model.astype(config.dtype_np_real),
            "radial_model": cov_radial_model.astype(config.dtype_np_real),
        },
        "ell_x_grid": ell_x_grid.astype(config.dtype_np_real),
        "ell_y_grid": ell_y_grid.astype(config.dtype_np_real),
        "ell_radial": ell_radial.astype(config.dtype_np_real),
        "ell_max": float(getattr(config, "ellmax", float("nan"))) if getattr(config, "ellmax", None) is not None else None,
        "metadata": {
            "n_src": int(n_src),
            "ny": int(ny),
            "nx": int(nx),
            "bands": list(config.bands),
            "description": PARAMETRIC_PRECISION_DESCRIPTION,
        },
    }


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


class ParametricPrecisionCalculator(NoisePSDCalculator):
    """Parametric precision calculator built into the production codebase."""

    def __init__(self, config, map_shape, precomputed_payload: Optional[Dict[str, Any]] = None):
        super().__init__(config, map_shape)
        self._payload: Optional[Dict[str, Any]] = None
        if precomputed_payload is not None:
            self._payload = self._normalize_payload(precomputed_payload)

    def calculate_noise_psd(self, maps_numpy: np.ndarray) -> np.ndarray:
        """
        Return the parametric precision matrix, computing it if necessary.

        Parameters
        ----------
        maps_numpy : np.ndarray
            Raw per-source maps with shape (n_src, ny, nx, n_bands, 3) in G3 units.

        Returns
        -------
        np.ndarray
            Precision tensor with shape (n_src, ny, nx, n_bands, 3, n_bands, 3).
        """
        if self._payload is None:
            logging.info("No cached parametric precision payload found; computing from scratch.")
            self._payload = self._normalize_payload(compute_parametric_precision(self.config, maps_numpy))
        precision = np.asarray(self._payload["precision"]).astype(self.config.dtype_np_complex)
        return precision

    @property
    def payload(self) -> Optional[Dict[str, Any]]:
        """Return the cached payload (precision + diagnostics) if available."""
        return self._payload

    def _normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a payload containing numpy arrays with consistent dtypes."""
        normalized: Dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                normalized[key] = {
                    sub_key: np.asarray(sub_val) if isinstance(sub_val, np.ndarray) else sub_val for sub_key, sub_val in value.items()
                }
            elif isinstance(value, np.ndarray):
                normalized[key] = np.asarray(value)
            else:
                normalized[key] = value
        return normalized


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
    elif config.noise_psd_method == "parametric_precision":
        return ParametricPrecisionCalculator(config, map_shape)
    elif config.noise_psd_method == "pca_psd_separate_tqu":
        return PcaPsdSeparateTQUCalculator(config, map_shape)
    elif config.noise_psd_method == "white_noise":
        return WhiteNoiseCalculator(config, map_shape)
    else:
        raise ValueError(f"Unknown noise_psd_method: {config.noise_psd_method}")
