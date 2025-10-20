"""
Noise Power Spectral Density (PSD) implementations for polarized beam fitting.

This module provides different approaches to estimating noise PSDs used in the
maximum-likelihood fitting of polarized beams.

There are several approaches for calculating the noise PSD. One of the major
differences is whether the noise PSD is fully diagonal (each ky,kx,band,stokes
is separate) or only diagonal in Fourier space (ky,ky independent, but band-band
and stokes-stokes off-diagonals).
We will use config.noise_psd_method to decide.
Currently, [clusterfinder_psd, kx_averaged, white_noise, ensemble_asd_mean, pca_psd, pca_psd_separate_tqu]
are fully diagonal in (band, stokes). [multiband_covariance, cmb_pca_perfield]
are not, as they model band and/or Stokes correlations.
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

PARAMETRIC_PRECISION_SUPERSAMPLING = 4
PARAMETRIC_PRECISION_HIGH_KY_THRESHOLD = 3000.0
PARAMETRIC_PRECISION_WHITE_NOISE_FLOOR_ELL = 8000.0
PARAMETRIC_PRECISION_RADIAL_MAX_ELL = 3000.0
PARAMETRIC_PRECISION_RADIAL_PIVOT = 1000.0
PARAMETRIC_PRECISION_RADIAL_EXPONENT = 1.0
PARAMETRIC_PRECISION_RADIAL_DC_FAKE = 50.0
PARAMETRIC_PRECISION_EIGEN_EPS = 1e-6
PARAMETRIC_PRECISION_ELL_X_PIVOT = 1000.0
PARAMETRIC_PRECISION_ELL_X_DC_FAKE = 50.0
PARAMETRIC_PRECISION_ELL_X_BOUNDS_LOWER = np.array([0.0, 0.0, 0.0])
PARAMETRIC_PRECISION_ELL_X_BOUNDS_UPPER = np.array([5e-7, 3e-4, 2e-7])
PARAMETRIC_PRECISION_DESCRIPTION = (
    "Precision matrix from parametric modelling: CMB (fixed calibration) + ell_x-dependent + radial uncorrelated model. "
    "Uses unshifted FFT convention (DC located at [0,0]) and discrete Fourier normalization."
)

# Tcal https://sptlocal.grid.uchicago.edu/~yomori/20192020_lensing/Tcal/v3/spt3g20192020_tcal.html#updated-calibration
# Pcal https://pole.uchicago.edu/spt3g/index.php/File:20231009_EETETT_Updates.pdf
CMB_CALIBRATION_FACTORS = np.array(
    [
        [1.07, 1.02, 1.01],
        [1.05, 1.06, 1.17],
        [1.05, 1.06, 1.17],
    ]
)


def _band_suffix(band_name: str) -> str:
    """Return a compact suffix (e.g. '150') extracted from a band label such as '150GHz'."""
    digits = "".join(ch for ch in band_name if ch.isdigit())
    return digits or band_name


def _effective_solid_angle(mask: np.ndarray, reso_arcmin: float) -> float:
    """Return the effective solid angle Ω_eff = Ω_pix * ∑ mask^2 in steradians."""
    dtheta_rad = reso_arcmin * (np.pi / (180.0 * 60.0))
    omega_pix = dtheta_rad**2
    return float(np.sum(mask**2)) * omega_pix


def _format_rms_value(value: float) -> str:
    """Return a readable string for µK-arcmin RMS values across several orders of magnitude."""
    abs_val = abs(value)
    if abs_val >= 1000.0:
        return f"{value:,.0f}"
    if abs_val >= 100.0:
        return f"{value:,.1f}"
    if abs_val >= 10.0:
        return f"{value:.2f}"
    if abs_val >= 0.1:
        return f"{value:.3f}"
    return f"{value:.2e}"


def _compute_rms_table(
    covariance: np.ndarray,
    reso_arcmin: float,
    bands: List[str],
) -> List[Tuple[str, float]]:
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

    for band_idx, band in enumerate(bands):
        suffix = _band_suffix(band)
        for stokes_idx, st_label in enumerate(stokes_labels):
            diag_vals = diag_view[..., band_idx, stokes_idx, band_idx, stokes_idx]
            diag_vals = np.clip(diag_vals, 0.0, None)
            rms_mK = float(np.sqrt(np.mean(diag_vals)))
            rms_uk_arcmin = rms_mK * 1000.0 * reso_arcmin
            rms_entries.append((f"{st_label}{suffix}", rms_uk_arcmin))
    return rms_entries


def _print_rms_summary(title: str, covariance: np.ndarray, config) -> None:
    """Log RMS summaries (in µK-arcmin) for each band/stokes diagonal of the provided covariance."""
    entries = _compute_rms_table(covariance, config.reso_arcmin, list(config.bands))
    if not entries:
        return
    formatted = ", ".join(f"{label}={_format_rms_value(value)} µK-arcmin" for label, value in entries)
    print(f"{title}: {formatted}")


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


def _project_to_spd(matrix: np.ndarray, eps: float = PARAMETRIC_PRECISION_EIGEN_EPS) -> np.ndarray:
    """Project a covariance matrix to the nearest symmetric positive-definite matrix."""
    symm = 0.5 * (matrix + matrix.T)
    diag = np.diag(symm)
    positive_diag = diag[diag > 0]
    assert positive_diag.size > 0, "No positive diagonal elements found in covariance matrix"
    scale = np.median(positive_diag)
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
    fft = np.fft.fft2(masked_maps, axes=(1, 2))

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
    print("Computing CAMB spectra for parametric precision model...")
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.1200, mnu=0.06, omk=0, tau=0.0544)
    pars.InitPower.set_params(As=2.1e-9, ns=0.9649)
    ell_max_cmb = 5000
    pars.set_for_lmax(ell_max_cmb, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)
    ell_camb = np.arange(powers["total"].shape[0])
    cl_tt_raw = powers["total"][:, 0] * 1e-6
    cl_ee_raw = powers["total"][:, 1] * 1e-6
    cl_bb_raw = powers["total"][:, 2] * 1e-6
    cl_te_raw = powers["total"][:, 3] * 1e-6

    print("Applying high-pass filter to CAMB spectra...")

    hp_filter = _smooth_highpass_1d(ell_camb, 360.0, 720.0)
    cl_tt = cl_tt_raw * hp_filter
    cl_ee = cl_ee_raw * hp_filter
    cl_bb = cl_bb_raw * hp_filter
    cl_te = cl_te_raw * hp_filter

    ny_highres = ny * PARAMETRIC_PRECISION_SUPERSAMPLING
    nx_highres = nx * PARAMETRIC_PRECISION_SUPERSAMPLING

    dell_x = ell_x_grid[0, 1] - ell_x_grid[0, 0]
    dell_y = ell_y_grid[1, 0] - ell_y_grid[0, 0]
    print(f"ell_x spacing = {dell_x:.2f}$, ell_y spacing = {dell_y:.2f}$")

    print("Interpolating CMB spectra onto high-resolution grid...")

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

    print("Converting from TEB to TQU...")

    phi = np.arctan2(ell_y_grid_highres, ell_x_grid_highres)
    c2phi = np.cos(2.0 * phi)
    s2phi = np.sin(2.0 * phi)

    print("Calculating CMB covariance on high-resolution grid...")
    cov_tqu_highres = np.zeros((ny_highres, nx_highres, 3, 3), dtype=config.dtype_np_real)
    cov_tqu_highres[..., 0, 0] = C_TT
    cov_tqu_highres[..., 0, 1] = cov_tqu_highres[..., 1, 0] = C_TE * c2phi
    cov_tqu_highres[..., 0, 2] = cov_tqu_highres[..., 2, 0] = C_TE * s2phi
    cov_tqu_highres[..., 1, 1] = C_EE * c2phi**2 + C_BB * s2phi**2
    cov_tqu_highres[..., 2, 2] = C_EE * s2phi**2 + C_BB * c2phi**2
    cov_tqu_highres[..., 1, 2] = cov_tqu_highres[..., 2, 1] = (C_EE - C_BB) * s2phi * c2phi

    print("Downsampling CMB covariance...")
    ss = PARAMETRIC_PRECISION_SUPERSAMPLING
    cov_tqu_highres_shifted = np.fft.fftshift(cov_tqu_highres)
    cov_tqu_highres_shifted = 0.5 * np.roll(cov_tqu_highres_shifted, (-1, -1), axis=(0, 1)) + 0.5 * cov_tqu_highres_shifted
    cov_tqu_shifted = cov_tqu_highres_shifted.reshape(ny, ss, nx, ss, 3, 3).mean(axis=(1, 3))
    cov_tqu = np.fft.ifftshift(cov_tqu_shifted)

    print("Building apod mask for CMB covariance convolution...")
    apod_mask = make_apodization_mask((config.map_size_pix, config.map_size_pix), config.apodization_width_pix)
    apod_mask_fft = np.fft.fft2(apod_mask)
    window_full = np.abs(apod_mask_fft) ** 2 / config.map_size_pix**2
    inds_y = np.concatenate((np.arange(0, ny // 2 + 1), np.arange(-ny // 2 + 1, 0)))
    inds_x = np.concatenate((np.arange(0, nx // 2 + 1), np.arange(-nx // 2 + 1, 0)))
    fftwindow_full = np.fft.fft2(window_full)
    fftwindow = fftwindow_full[np.ix_(inds_y, inds_x)]
    fft_cov_tqu = np.fft.fft2(cov_tqu, axes=(0, 1))
    fft_cov_tqu_convolved = fft_cov_tqu * fftwindow[:, :, None, None]
    cov_tqu_convolved = np.fft.ifft2(fft_cov_tqu_convolved, axes=(0, 1))

    print("Applying CMB calibration factors...")
    cov_cmb = np.zeros((ny, nx, n_bands, 3, n_bands, 3), dtype=config.dtype_np_real)
    for iband in range(n_bands):
        for istokes in range(3):
            for jband in range(n_bands):
                for jstokes in range(3):
                    cov_cmb[:, :, iband, istokes, jband, jstokes] = (
                        cov_tqu_convolved[:, :, istokes, jstokes]
                        * CMB_CALIBRATION_FACTORS[istokes, iband]
                        * CMB_CALIBRATION_FACTORS[jstokes, jband]
                    )

    # something is wrong with the normalization (about a factor of 3000...)
    # might need some of the following
    # noise_mask = make_apod_mask_center_excised(
    #     (config.map_size_pix, config.map_size_pix),
    #     config.apodization_width_pix,
    #     config.noise_hole_radius_arcmin,
    #     config.reso_arcmin,
    # )
    # dtheta_rad = config.reso_arcmin * (np.pi / (180.0 * 60.0)) # 3e-5
    # Omega_pix = dtheta_rad**2 # 8e-10
    # Omega_eff = Omega_pix * float(np.sum(noise_mask**2)) # 8e-5

    return cov_cmb


def _design_matrix_ell_x(ell_x_vals: np.ndarray) -> np.ndarray:
    """Return the simplified design matrix for ell_x fitting with three basis functions."""
    ell_x_vals = np.asarray(ell_x_vals, dtype=float)
    pivot = PARAMETRIC_PRECISION_ELL_X_PIVOT
    design = np.zeros((len(ell_x_vals), 3), dtype=float)
    design[:, 0] = (pivot / ell_x_vals) ** 2
    design[:, 1] = 1.0
    design[:, 2] = (ell_x_vals / pivot) ** 2
    return design


def _fit_ell_x_model(
    cov_no_cmb: np.ndarray,
    cov_input_for_floor: np.ndarray,
    ell_x: np.ndarray,
    high_ky_mask: np.ndarray,
) -> np.ndarray:
    """Fit the ell_x-dependent uncorrelated noise model using log-space weighted least squares."""
    n_src, _, nx, n_bands, n_stokes, _, _ = cov_no_cmb.shape

    ell_x_fit = ell_x.copy()
    ell_x_fit[ell_x == 0.0] = PARAMETRIC_PRECISION_ELL_X_DC_FAKE

    design = _design_matrix_ell_x(ell_x_fit)
    cov_ellx = np.zeros((n_src, nx, n_bands, n_stokes, n_bands, n_stokes), dtype=cov_no_cmb.dtype)

    if not np.any(high_ky_mask):
        print("High-k_y mask for ell_x fitting is empty; falling back to full grid.")
        averaged = np.mean(cov_no_cmb, axis=1)
    else:
        averaged = np.mean(cov_no_cmb[:, high_ky_mask, :, :, :, :, :], axis=1)

    for src in range(n_src):
        for band in range(n_bands):
            for stokes in range(n_stokes):
                y = averaged[src, :, band, stokes, band, stokes]
                y = np.maximum(y, 1e-20)

                weights = 1.0 / y
                W = np.sqrt(weights)
                W_diag = np.diag(W)
                design_weighted = W_diag @ design
                y_weighted = W_diag @ y

                theta, *_ = np.linalg.lstsq(design_weighted, y_weighted, rcond=None)
                theta = np.clip(theta, PARAMETRIC_PRECISION_ELL_X_BOUNDS_LOWER, PARAMETRIC_PRECISION_ELL_X_BOUNDS_UPPER)

                model = design @ theta

                white_mask = np.abs(ell_x) >= PARAMETRIC_PRECISION_WHITE_NOISE_FLOOR_ELL
                if np.any(white_mask):
                    reference = cov_input_for_floor[src, :, :, band, stokes, band, stokes]
                    if reference.size:
                        floor = 0.7 * np.median(reference[white_mask])
                        model = np.maximum(model, floor)

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
        print("Radial basis denominator non-positive; skipping radial fit.")
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


def compute_parametric_precision(
    config,
    raw_maps: np.ndarray,
    source_fields: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
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
    print("Starting parametric precision computation...")

    maps_numpy = np.asarray(raw_maps, dtype=config.dtype_np_real)
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
    print("Inverting covariance model to precision matrices...")
    for src in range(n_src):
        for iy in range(ny):
            for ix in range(nx):
                cov_9x9 = cov_model_full[src, iy, ix].reshape(n_dim, n_dim)
                cov_spd = _project_to_spd(cov_9x9, PARAMETRIC_PRECISION_EIGEN_EPS)
                try:
                    prec = np.linalg.inv(cov_spd)
                except np.linalg.LinAlgError:
                    print(f"Singular covariance matrix at src {src}, iy {iy}, ix {ix}. Using pseudo-inverse.")
                    prec = np.linalg.pinv(cov_spd)
                prec = 0.5 * (prec + prec.T)
                precision[src, iy, ix] = prec.reshape(n_bands, 3, n_bands, 3)

    precision *= Omega_pix
    perfield_median_applied = False
    metadata_fields = None
    if getattr(config, "parametric_precision_perfield_median", False):
        if source_fields is None:
            raise ValueError(
                "parametric_precision_perfield_median=True requires per-source field labels. "
                "Provide them via PrecisionCalculator.set_source_fields before computing precision."
            )
        source_fields = np.asarray(source_fields)
        if source_fields.shape[0] != n_src:
            raise ValueError(f"Length of provided source_fields does not match number of sources ({source_fields.shape[0]} != {n_src}).")
        unique_fields = np.unique(source_fields)
        print("Applying per-field median pooling to parametric precision (fields: %s).", unique_fields.tolist())
        for field in unique_fields:
            field_mask = source_fields == field
            if not np.any(field_mask):
                continue
            field_precision = precision[field_mask]
            median_precision = np.median(field_precision, axis=0)
            median_precision = np.asarray(median_precision, dtype=config.dtype_np_complex)
            precision[field_mask] = median_precision
        metadata_fields = [None if field is None else str(field) for field in source_fields]
        perfield_median_applied = True

    print("Parametric precision computation complete.")
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
            "parametric_precision_perfield_median": perfield_median_applied,
            "source_fields": metadata_fields,
        },
    }


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

    def _diagonal_psd_to_precision(self, psd: np.ndarray, n_src: int) -> np.ndarray:
        """Return per-source precision grids for diagonal PSD models."""
        psd_array = np.asarray(psd)
        if psd_array.ndim == 4:
            psd_array = np.broadcast_to(psd_array, (n_src,) + psd_array.shape)
        if psd_array.ndim != 5:
            raise ValueError(f"Unexpected PSD shape {psd_array.shape}; expected (..., ny, nx, n_bands, 3).")
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.reciprocal(psd_array)
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

    def _invert_covariance_stack(self, covariance_psd: np.ndarray) -> np.ndarray:
        """Invert a single covariance grid with axes (ny, nx, n_bands, 3, n_bands, 3)."""
        ny, nx, n_bands, n_stokes, _, _ = covariance_psd.shape
        n_dim = n_bands * n_stokes
        assert covariance_psd.dtype == self.config.dtype_np_real, (
            f"Covariance must be {self.config.dtype_np_real}, not {covariance_psd.dtype}"
        )
        precision = np.zeros((ny, nx, n_bands, n_stokes, n_bands, n_stokes), dtype=self.config.dtype_np_real)
        for iy in range(ny):
            for ix in range(nx):
                cov = covariance_psd[iy, ix].reshape(n_dim, n_dim)
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

        precision = self._diagonal_psd_to_precision(noise_psd_array, maps_numpy.shape[0])
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
    Calculate individual noise PSDs using k_x averaging with max heuristic,
    then average over all sources.

    This implementation estimates noise by analyzing regions of each map that are
    away from the central source, then averages over k_y for each k_x mode and
    takes the element-wise maximum with the original PSD to avoid scattered low values.
    """

    def calculate_precision(
        self,
        maps_numpy: np.ndarray,
        idx_y: Optional[np.ndarray] = None,
        idx_x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Estimate a diagonal precision model via kx-averaged PSD heuristics."""
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
        precision = self._diagonal_psd_to_precision(global_noise_psd, maps_numpy.shape[0])
        precision = self._truncate_fourier_numpy(precision, idx_y, idx_x, axis_y=1, axis_x=2)
        return precision, None

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

        # Take FFT and calculate power spectral density in flat-sky normalization
        dtheta_rad = self.config.reso_arcmin * (np.pi / (180.0 * 60.0))
        omega_pix = dtheta_rad**2
        fft_2d = np.fft.fft2(masked_map) * omega_pix
        psd_2d = np.abs(fft_2d) ** 2

        # Normalize by the effective solid angle (mask power × pixel area)
        omega_eff = _effective_solid_angle(noise_mask, self.config.reso_arcmin)
        dtheta_rad = self.config.reso_arcmin * (np.pi / (180.0 * 60.0))
        omega_pix = dtheta_rad**2
        psd_2d /= omega_eff
        psd_2d /= omega_pix

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


class EnsembleAsdMeanCalculator(PrecisionCalculator):
    """
    Calculate PSDs by averaging amplitude spectral densities across sources.

    This implementation takes the PSD of each source (with center-excised apodization),
    converts to amplitude spectral density (ASD), averages across all sources,
    then converts back to PSD.
    """

    def calculate_precision(
        self,
        maps_numpy: np.ndarray,
        idx_y: Optional[np.ndarray] = None,
        idx_x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Estimate diagonal precision via ensemble-averaged ASDs."""
        print("Calculating ensemble-averaged ASD-derived PSDs...")

        # Create noise mask with hole in center to avoid the source signal
        noise_mask = make_apod_mask_center_excised(
            self.map_shape,
            self.config.apodization_width_pix,
            self.config.noise_hole_radius_arcmin,
            self.config.reso_arcmin,
        )

        dtheta_rad = self.config.reso_arcmin * (np.pi / (180.0 * 60.0))
        omega_pix = dtheta_rad**2

        # Collect ASDs from all sources
        ny, nx = self.map_shape
        n_src = maps_numpy.shape[0]
        all_asds = np.zeros((n_src, ny, nx, self.n_bands, 3), dtype=self.config.dtype_np_real)

        omega_eff = _effective_solid_angle(noise_mask, self.config.reso_arcmin)

        for i in range(n_src):
            print(f"  Processing source {i + 1}/{n_src}")

            for band_idx, band in enumerate(self.config.bands):
                for stokes_idx, stokes in enumerate(["T", "Q", "U"]):
                    real_map = maps_numpy[i, :, :, band_idx, stokes_idx]

                    # Apply center-excised mask
                    masked_map = real_map * noise_mask

                    # Calculate PSD and convert to ASD
                    fft_2d = np.fft.fft2(masked_map) * omega_pix
                    psd_2d = np.abs(fft_2d) ** 2

                    # Normalize by effective solid angle and pixel area
                    psd_2d /= omega_eff
                    psd_2d /= omega_pix

                    # Convert PSD to ASD (amplitude spectral density)
                    asd_2d = np.sqrt(psd_2d)
                    all_asds[i, :, :, band_idx, stokes_idx] = asd_2d

        # Average ASDs across all sources and convert back to PSD
        print("  Averaging ASDs across sources...")
        mean_asd = np.mean(all_asds, axis=0)  # Average over sources
        mean_psd = mean_asd**2

        print("Ensemble ASD averaging complete.")
        precision = self._diagonal_psd_to_precision(mean_psd, maps_numpy.shape[0])
        precision = self._truncate_fourier_numpy(precision, idx_y, idx_x, axis_y=1, axis_x=2)
        return precision, None


class MultiBandCovarianceCalculator(PrecisionCalculator):
    """
    Calculate multi-band covariance PSD for simultaneous fitting across frequency bands.

    This implementation creates a (ky,kx,band,band,stokes,stokes) covariance matrix capturing correlations
    between bands and Stokes parameters (T,Q,U), computed using center-excised apodization and averaged across sources.
    """

    def __init__(self, config, map_shape):
        super().__init__(config, map_shape)
        # Use the bands from config instead of hard-coding
        # They are already sorted by the parent class

    def calculate_precision(
        self,
        maps_numpy: np.ndarray,
        idx_y: Optional[np.ndarray] = None,
        idx_x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Compute per-field multi-band covariance models and invert them to precision."""
        noise_mask = make_apod_mask_center_excised(
            self.map_shape,
            self.config.apodization_width_pix,
            self.config.noise_hole_radius_arcmin,
            self.config.reso_arcmin,
        )
        omega_eff = _effective_solid_angle(noise_mask, self.config.reso_arcmin)
        dtheta_rad = self.config.reso_arcmin * (np.pi / (180.0 * 60.0))
        omega_pix = dtheta_rad**2

        def compute(field_maps: np.ndarray, field, indices):
            print(f"Calculating multi-band covariance PSD for field {field if field is not None else 'all'}...")
            masked_maps = field_maps * noise_mask[None, :, :, None, None]
            masked_maps_fft = np.fft.fft2(masked_maps, axes=(1, 2)) * omega_pix
            n_field_src = masked_maps_fft.shape[0]
            covariance_sum = np.einsum("nyxbs,nyxct->yxbcst", masked_maps_fft, np.conj(masked_maps_fft))
            covariance_psd = covariance_sum / (n_field_src * omega_eff * omega_pix)
            covariance_psd = np.transpose(covariance_psd, (0, 1, 2, 4, 3, 5))
            covariance_psd = covariance_psd.astype(self.config.dtype_np_complex, copy=False)
            covariance_psd = self._truncate_fourier_numpy(covariance_psd, idx_y, idx_x, axis_y=0, axis_x=1)
            precision_grid = self._covariance_to_precision(covariance_psd)
            broadcast = np.broadcast_to(precision_grid, (len(indices),) + precision_grid.shape)
            print(f"Multi-band covariance calculation complete using {n_field_src} sources.")
            debug_field = {"covariance": covariance_psd}
            return broadcast.astype(self.config.dtype_np_complex, copy=False), debug_field

        precision, debug = self._apply_fieldwise(maps_numpy, compute)
        return precision, (debug or None)


class PcaMultiBandCalculator(PrecisionCalculator):
    """PCA-regularized multi-band precision estimator."""

    def calculate_precision(
        self,
        maps_numpy: np.ndarray,
        idx_y: Optional[np.ndarray] = None,
        idx_x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        def compute(field_maps: np.ndarray, field, indices):
            print(
                f"Calculating PCA-based multi-band precision ({self.config.n_pca_components} components) "
                f"for field {field if field is not None else 'all'}..."
            )
            n_src, ny, nx, n_bands, n_stokes = field_maps.shape
            n_dim = n_bands * n_stokes

            noise_mask = make_apod_mask_center_excised(
                self.map_shape,
                self.config.apodization_width_pix,
                self.config.noise_hole_radius_arcmin,
                self.config.reso_arcmin,
            )

            masked_maps = field_maps * noise_mask[None, :, :, None, None]
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
                raise ValueError(
                    "n_pca_components must be positive and strictly less than the number of sources for PCA precision estimation."
                )

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
                (components_real + 1j * components_imag)
                .reshape(n_components, ny, nx, n_bands, n_stokes)
                .astype(self.config.dtype_np_complex)
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

            precision_matrix = self._truncate_fourier_numpy(precision_matrix, idx_y, idx_x, axis_y=0, axis_x=1)
            broadcast = np.broadcast_to(precision_matrix, (len(indices),) + precision_matrix.shape)
            print("PCA-based multi-band precision calculation complete.")
            return broadcast.astype(self.config.dtype_np_complex, copy=False), None

        precision, debug = self._apply_fieldwise(maps_numpy, compute)
        return precision, (debug or None)


class ParametricPrecisionCalculator(PrecisionCalculator):
    """Parametric precision calculator built into the production codebase."""

    def __init__(self, config, map_shape, precomputed_payload: Optional[Dict[str, Any]] = None):
        super().__init__(config, map_shape)
        self._payload: Optional[Dict[str, Any]] = None
        if precomputed_payload is not None:
            self._payload = self._normalize_payload(precomputed_payload)

    def calculate_precision(
        self,
        maps_numpy: np.ndarray,
        idx_y: Optional[np.ndarray] = None,
        idx_x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
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
            print("No cached parametric precision payload found; computing from scratch.")
            self._payload = self._normalize_payload(compute_parametric_precision(self.config, maps_numpy, source_fields=self.source_fields))
        precision = np.asarray(self._payload["precision"]).astype(self.config.dtype_np_complex)
        precision = self._truncate_fourier_numpy(precision, idx_y, idx_x, axis_y=1, axis_x=2)
        return precision, (self._payload if self.config.debug else None)

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


class CmbPcaPerFieldCalculator(PrecisionCalculator):
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
        dtheta_rad = self.config.reso_arcmin * (np.pi / (180.0 * 60.0))
        omega_pix = dtheta_rad**2
        omega_eff = _effective_solid_angle(noise_mask, self.config.reso_arcmin)

        if idx_y is None or idx_x is None:
            idx_y, idx_x = _ensure_fft_cut_indices((ny_full, nx_full), self.config)
        idx_y_eff, idx_x_eff = idx_y, idx_x
        sliced_to_indices = idx_y_eff is not None and idx_x_eff is not None
        ky_full, kx_full, ky_grid_full, kx_grid_full = compute_fourier_frequency_axes((ny_full, nx_full), self.config.reso_arcmin)

        masked_maps = maps_numpy * noise_mask[None, :, :, None, None]
        fft_maps = np.fft.fft2(masked_maps, axes=(1, 2)) * omega_pix

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

        covariance = np.einsum("nyxbs,nyxct->nyxbsct", fft_maps, np.conj(fft_maps), optimize=True).real.astype(
            self.config.dtype_np_real
        ) / (omega_eff * omega_pix)

        print(f"Largest ell_x value in covariance: {float(np.max(ell_x_grid)):.2f}")
        _print_rms_summary("Average noise (all ell modes)", covariance, self.config)

        ny, nx = ell_x_grid.shape
        print("Calculating expected CMB covariance...")
        cov_cmb = _compute_cmb_covariance(self.config, ny, nx, ell_x_grid, ell_y_grid, ell_radial, n_bands)
        cov_cmb = cov_cmb / omega_pix
        cov_cmb *= 1e-6  # Convert CAMB µK^2 spectrum to mK^2 in our discrete normalization.
        _print_rms_summary("Average CMB noise (all ell modes)", cov_cmb[None, ...], self.config)

        cov_no_cmb = covariance - cov_cmb[None, :, :, :, :, :, :]
        _print_rms_summary("Average noise (after CMB subtraction)", cov_no_cmb, self.config)

        diag_original = self._extract_diagonals(cov_no_cmb)
        print("Applying PCA to regularize band-stokes diagonal noise PSDs...")
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
        _print_rms_summary("Average noise (after PCA regularization)", residual, self.config)
        _print_rms_summary("Average noise (after PCA regularization and adding CMB)", covariance_model, self.config)
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
            print("cmb_pca_perfield white-noise region mask is empty; falling back to full grid for floor estimation.")
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
                floor_rms = float(np.sqrt(floor)) * 1000.0 * self.config.reso_arcmin
                print(f"White noise floor {label}: {_format_rms_value(floor_rms)} µK-arcmin")
                below_floor_mask = bounded[:, :, :, band, stokes] < floor
                if np.any(below_floor_mask):
                    below_floor_fraction = np.sum(below_floor_mask) / np.prod(below_floor_mask.shape)
                    print(
                        f"Some PSD values are below the white noise floor {label}. {below_floor_fraction:.2%} of the values are below the floor. Clipping to the floor and continuing..."
                    )
                    bounded[:, :, :, band, stokes][below_floor_mask] = floor
        return bounded


class PcaPsdSeparateTQUCalculator(PrecisionCalculator):
    """
    PcaPsdSeparateTQUCalculator performs separate PCA analyses for each Stokes parameter:
    one for temperature (T), one for Q polarization, and one for U polarization.

    This approach allows for different noise structures between temperature and each
    polarization component, which can be important for cosmic microwave background observations.
    """

    def calculate_precision(
        self,
        maps_numpy: np.ndarray,
        idx_y: Optional[np.ndarray] = None,
        idx_x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
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
        precision = self._diagonal_psd_to_precision(per_source_psd_array, maps_numpy.shape[0])
        precision = self._truncate_fourier_numpy(precision, idx_y, idx_x, axis_y=1, axis_x=2)
        return precision, None


class WhiteNoiseCalculator(PrecisionCalculator):
    """
    Calculate simple white noise PSD with constant values.

    This implementation assumes white noise with constant PSD values
    across all k-space for testing and baseline comparisons.
    """

    def calculate_precision(
        self,
        maps_numpy: np.ndarray,
        idx_y: Optional[np.ndarray] = None,
        idx_x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        print("Generating simple white noise PSD...")

        ny, nx = self.map_shape

        # Create white noise PSD array with all ones
        noise_psd_array = np.ones((ny, nx, self.n_bands, 3), dtype=self.config.dtype_np_real)

        # Apply scaling for polarization
        noise_psd_array[:, :, :, 1] *= 2.0  # Q polarization has 2x the noise
        noise_psd_array[:, :, :, 2] *= 2.0  # U polarization has 2x the noise

        print("White noise PSD complete.")
        precision = self._diagonal_psd_to_precision(noise_psd_array, maps_numpy.shape[0])
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
    elif config.noise_psd_method == "cmb_pca_perfield":
        return CmbPcaPerFieldCalculator(config, map_shape)
    elif config.noise_psd_method == "white_noise":
        return WhiteNoiseCalculator(config, map_shape)
    else:
        raise ValueError(f"Unknown noise_psd_method: {config.noise_psd_method}")
