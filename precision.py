"""
Fourier-domain precision estimation for polarized beam fitting.

This module provides a single calculate_precision function that estimates
precision matrices for noise modeling in the discrete Fourier grid.
"""

from typing import Any, Dict, Optional, Tuple

import camb
import numpy as np

from .utils import (
    compute_rectangular_ell_cut_indices,
    ell_grid,
    make_apod_mask_center_excised,
    make_apodization_mask,
)

CMB_SUPERSAMPLING = 4
COVARIANCE_EIGEN_EPS = 1e-4

CMB_CORRELATION_MAX = 0.95


def _smooth_highpass_1d(ell_vals: np.ndarray, ell0: float = 360.0, ell1: float = 1080.0) -> np.ndarray:
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
    ny_highres = ny * ss
    nx_highres = nx * ss
    dtheta_rad = config.reso_arcmin * (np.pi / (180.0 * 60.0))
    map_length_rad = config.map_size_pix * dtheta_rad
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

    # Match spt3g.mapspectra.basicmaputils IAU angle convention directly:
    #   phi = atan2(-kx, ky)
    phi = np.arctan2(-ell_x_grid_highres, ell_y_grid_highres)
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

    ny_map, nx_map = config.map_size_pix, config.map_size_pix
    apod_mask = make_apodization_mask((ny_map, nx_map), config.apodization_width_pix)
    apod_mask_fft = np.fft.fft2(apod_mask)
    window_full = np.abs(apod_mask_fft) ** 2 / (ny_map * nx_map)
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
                    cal_i = 1.0 if config.use_cdrc else config.cmb_calibration_factors[stokes_i_label][band_i_label]
                    cal_j = 1.0 if config.use_cdrc else config.cmb_calibration_factors[stokes_j_label][band_j_label]
                    cov_cmb[:, :, band_i_idx, stokes_i_idx, band_j_idx, stokes_j_idx] = (
                        cov_tqu_convolved[:, :, stokes_i_idx, stokes_j_idx] / cal_i / cal_j
                    )

    cov_cmb /= pixel_area_sr
    return cov_cmb


def _compute_white_noise_floors(variances: np.ndarray, ell_y_grid: np.ndarray, ell_x_grid: np.ndarray) -> np.ndarray:
    """Compute white noise floors from high-ell regions."""
    ny, nx, n_bands, n_stokes = variances.shape
    region_mask = (np.abs(ell_y_grid) > 4000.0) & (np.abs(ell_x_grid) > 4000.0) & (np.abs(ell_x_grid) < 8000.0)
    assert np.any(region_mask), "White-noise region mask is empty."
    floors = np.zeros((n_bands, n_stokes), dtype=variances.dtype)
    for band_idx in range(n_bands):
        for stokes_idx in range(n_stokes):
            these_variances = variances[:, :, band_idx, stokes_idx]
            floors[band_idx, stokes_idx] = np.percentile(these_variances[region_mask], 20.0)
    assert np.all(floors > 0.0), "Invalid white noise floor."
    return floors


def _invert_covariance(covariance: np.ndarray, config) -> np.ndarray:
    """Invert a single covariance grid with axes (ny, nx, n_bands, 3, n_bands, 3)."""
    # check if off-diagonals are all zero
    ny, nx, n_bands, n_stokes, _, _ = covariance.shape
    precision = np.zeros_like(covariance, dtype=config.dtype_np_real)
    for y_idx in range(ny):
        for x_idx in range(nx):
            this_covariance = covariance[y_idx, x_idx].reshape(n_bands * n_stokes, n_bands * n_stokes)
            if np.allclose(this_covariance, np.diag(np.diag(this_covariance))):
                b_diag = np.arange(n_bands)[:, None]
                s_diag = np.arange(n_stokes)[None, :]
                precision[y_idx, x_idx, b_diag, s_diag, b_diag, s_diag] = (
                    1 / this_covariance[b_diag * n_stokes + s_diag, b_diag * n_stokes + s_diag]
                )
            else:
                n_dim = n_bands * n_stokes
                cov = this_covariance.reshape(n_dim, n_dim)
                cov_spd = _project_to_spd(cov)
                inv = np.linalg.inv(cov_spd)
                inv = 0.5 * (inv + inv.T)
                precision[y_idx, x_idx] = inv.reshape(n_bands, n_stokes, n_bands, n_stokes)

    return precision


def calculate_precision(maps: np.ndarray, config) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """
    Calculate precision matrices for noise modeling.

    Parameters
    ----------
    maps : np.ndarray
        Input maps with shape (n_src_field, ny_full, nx_full, n_bands, n_stokes)
    config : object
        BeamFittingConfig instance

    Returns
    -------
    precision : np.ndarray
        Precision array with shape (ny, nx, n_bands, n_stokes, n_bands, n_stokes)
    debug : dict or None
        Debug information if config.debug is True, None otherwise
    """
    assert config.precision_n_pca == 0, "PCA regularization is no longer supported."
    n_src, ny_full, nx_full, n_bands, n_stokes = maps.shape
    ell_y_full, ell_x_full, ell_y_grid_full, ell_x_grid_full = ell_grid((ny_full, nx_full), config.reso_arcmin)
    idx_y, idx_x = compute_rectangular_ell_cut_indices((ny_full, nx_full), config.reso_arcmin, config.ellmax)
    ell_y, ell_x = ell_y_full[idx_y], ell_x_full[idx_x]
    ell_y_grid, ell_x_grid = np.meshgrid(ell_y, ell_x, indexing="ij")
    ell_r_grid = np.sqrt(ell_x_grid**2 + ell_y_grid**2)
    ny, nx = len(ell_y), len(ell_x)

    n_bands = len(config.bands)
    n_stokes = 3

    noise_mask = make_apod_mask_center_excised(
        (ny_full, nx_full),
        config.apodization_width_pix,
        config.noise_hole_radius_arcmin,
        config.reso_arcmin,
    ).astype(config.dtype_np_real)

    noise_maps = maps * noise_mask[None, :, :, None, None]
    noise_maps_fft_full = np.fft.fft2(noise_maps, axes=(1, 2))
    noise_maps_fft_truncated_y = noise_maps_fft_full[:, idx_y, :, :, :]
    noise_maps_fft = noise_maps_fft_truncated_y[:, :, idx_x, :, :]
    covariance_data_full = np.einsum("nyxbs,nyxct->nyxbsct", noise_maps_fft, np.conj(noise_maps_fft), optimize=True).real
    covariance_data = np.mean(covariance_data_full, axis=0)
    if config.debug:
        debug = {
            "covariance_data": covariance_data,
            "ell_y_grid": ell_y_grid,
            "ell_x_grid": ell_x_grid,
            "ell_r_grid": ell_r_grid,
        }

    b_diag = np.arange(n_bands)[:, None]
    s_diag = np.arange(n_stokes)[None, :]
    variance_data = covariance_data[:, :, b_diag, s_diag, b_diag, s_diag]
    white_noise_floors = _compute_white_noise_floors(variance_data, ell_y_grid, ell_x_grid)
    print(f"White noise floors: {white_noise_floors}")
    covariance_model = np.zeros((ny, nx, n_bands, n_stokes, n_bands, n_stokes), dtype=config.dtype_np_real)

    if config.precision_white_noise:
        print("Creating white noise covariance...")
        for band_idx in range(n_bands):
            for stokes_idx in range(n_stokes):
                covariance_model[:, :, band_idx, stokes_idx, band_idx, stokes_idx] = white_noise_floors[band_idx, stokes_idx]
                covariance_model[0, 0, band_idx, stokes_idx, band_idx, stokes_idx] = 1e12  # kill the DC mode
    elif config.precision_model_cmb:
        # raise NotImplementedError("Model CMB covariance is currently broken. Needs to be debugged.")
        print("Creating covariance from combining average data covariance and expected CMB covariance...")
        covariance_cmb = _compute_cmb_covariance(config, ny, nx)
        for band_idx_i in range(n_bands):
            for stokes_idx_i in range(n_stokes):
                for band_idx_j in range(n_bands):
                    for stokes_idx_j in range(n_stokes):
                        if band_idx_i == band_idx_j and stokes_idx_i == stokes_idx_j:
                            # diagonals: take the element-wise maximum of (CMB + white noise floor) and data-mean
                            cmb_plus_white = (
                                covariance_cmb[:, :, band_idx_i, stokes_idx_i, band_idx_j, stokes_idx_j]
                                + white_noise_floors[band_idx_i, stokes_idx_i]
                            )
                            covariance_model[:, :, band_idx_i, stokes_idx_i, band_idx_j, stokes_idx_j] = np.maximum(
                                cmb_plus_white,
                                covariance_data[:, :, band_idx_i, stokes_idx_i, band_idx_j, stokes_idx_j],
                            )
                        else:
                            # off-diagonals: take the CMB covariance
                            covariance_model[:, :, band_idx_i, stokes_idx_i, band_idx_j, stokes_idx_j] = covariance_cmb[
                                :, :, band_idx_i, stokes_idx_i, band_idx_j, stokes_idx_j
                            ]
        # clip the correlation matrix
        for iy in range(ny):
            for ix in range(nx):
                this_covariance = covariance_model[iy, ix].reshape(n_bands * n_stokes, n_bands * n_stokes)
                this_covariance_symm = 0.5 * (this_covariance + this_covariance.T)

                # Build correlation matrix R = D^{-1/2} C D^{-1/2}
                d = np.diag(this_covariance_symm)
                dsqrt = np.sqrt(d)
                invdsqrt = 1.0 / dsqrt
                correlation = this_covariance_symm * (invdsqrt[:, None] * invdsqrt[None, :])
                correlation = np.clip(correlation, -CMB_CORRELATION_MAX, CMB_CORRELATION_MAX)
                np.fill_diagonal(correlation, 1.0)  # restore the unit diagonal
                this_covariance = correlation * (dsqrt[:, None] * dsqrt[None, :])
                covariance_model[iy, ix] = this_covariance.reshape(n_bands, n_stokes, n_bands, n_stokes)

    elif config.precision_datadriven_offdiagonals:
        print("Using average data covariance...")
        covariance_model[:] = covariance_data[:]

    else:
        print("Using average data variance...")
        for band_idx in range(n_bands):
            for stokes_idx in range(n_stokes):
                covariance_model[:, :, band_idx, stokes_idx, band_idx, stokes_idx] = variance_data[:, :, band_idx, stokes_idx]

    print("Clipping variances to white noise floors...")
    for band_idx_i in range(n_bands):
        for stokes_idx_i in range(n_stokes):
            below_floor = (
                covariance_model[:, :, band_idx_i, stokes_idx_i, band_idx_i, stokes_idx_i] < white_noise_floors[band_idx_i, stokes_idx_i]
            )
            if np.any(below_floor):
                print(f"{np.mean(below_floor) * 100:.2f}% of modes have variance lower than white noise floor. Setting to floor.")
                covariance_model[:, :, band_idx_i, stokes_idx_i, band_idx_i, stokes_idx_i][below_floor] = white_noise_floors[
                    band_idx_i, stokes_idx_i
                ]

    if config.debug:
        debug["covariance_model"] = covariance_model

    print("Inverting covariance matrices...")
    precision = _invert_covariance(covariance_model, config)

    if config.debug:
        debug["precision"] = precision
    else:
        debug = None

    return precision, debug
