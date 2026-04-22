"""
Source fitting functions for polarized beam analysis.

Contains functions for Gaussian fitting of point sources in T, Q, U maps.
"""

import numpy as np
from scipy import interpolate, optimize

ARC_MIN_RAD = np.pi / (180.0 * 60.0)
DEG_RAD = np.pi / 180.0


def iso_hpf_transfn(sigma, ell_cutoff, ell):
    """
    Return the isotropic high-pass transfer function used for the wafer common-mode approximation.

    Above ``ell_cutoff`` the transfer function is 1. Below it, the response is
    a Gaussian centered on ``ell_cutoff`` with width ``sigma``.
    """
    hpf = np.exp(-((ell - ell_cutoff) ** 2) / (2 * sigma**2))
    hpf[ell > ell_cutoff] = 1.0
    return hpf


def highpass_func(f, cutoff_freq):
    """SPT3G-compatible one-dimensional high-pass response."""
    out = np.zeros_like(f, dtype=float)
    out[f != 0] = np.exp(-1.0 * (cutoff_freq / f[f != 0]) ** 6.0)
    return out


def lowpass_func(f, cutoff_freq):
    """SPT3G-compatible one-dimensional low-pass response."""
    return np.exp(-1.0 * (f / cutoff_freq) ** 6.0)


def get_lxly(mapres, mapshape):
    """Return full-complex FFT-order ell_x and ell_y grids for a flat map."""
    ly_1d = np.fft.fftfreq(mapshape[0], mapres) * 2.0 * np.pi
    lx_1d = np.fft.fftfreq(mapshape[1], mapres) * 2.0 * np.pi
    return np.meshgrid(lx_1d, ly_1d)


def construct_tf(mapshape, mapres, ellhpf, elllpf, cm="wafer"):
    """
    Construct the source-fitting transfer function used for filtered Gaussian templates.

    The function combines an x-direction high-pass, x-direction low-pass, and
    approximate by-wafer common-mode response. ``mapres`` is in radians.
    """
    ell_x, ell_y = get_lxly(mapres, mapshape)
    if cm == "wafer":
        cm_tf = iso_hpf_transfn(240, 755, np.sqrt(ell_x**2 + ell_y**2))
    elif cm is None or not cm:
        cm_tf = np.ones(mapshape)
    else:
        raise NotImplementedError("Only by-wafer common mode is implemented")
    hpf_tf = highpass_func(ell_x, ellhpf)
    lpf_tf = lowpass_func(ell_x, elllpf)
    return cm_tf * hpf_tf * lpf_tf


def make_filtered_gaussian_template(sigma, res=0.1 * ARC_MIN_RAD, ret_array=False):
    """
    Construct the filtered Gaussian point-source template used by the source fits.

    Parameters
    ----------
    sigma : float
        Gaussian beam width in radians.
    res : float
        Pixel resolution in radians.
    ret_array : bool
        Return the template array instead of a ``RectBivariateSpline``.
    """
    side = DEG_RAD
    npix = int(np.ceil(side / res))
    center_pix = (npix - 1) / 2

    y, x = (np.mgrid[:npix, :npix] - center_pix) * res
    gaussian = np.exp(-0.5 * (x**2 + y**2) / sigma**2)
    gauss_ft = np.fft.fft2(gaussian)
    filt_ft = gauss_ft * construct_tf(gaussian.shape, res, 500, 20000)
    tmpl = np.fft.ifft2(filt_ft)
    if not np.allclose(np.imag(tmpl), 0):
        raise RuntimeError("Filtered Gaussian template has a non-negligible imaginary component.")
    tmpl = np.real(tmpl)
    if ret_array:
        return tmpl
    grid = np.arange(-center_pix, center_pix + 1) * res
    return interpolate.RectBivariateSpline(grid, grid, tmpl)


def err_func(pars, mapunw, invnoise, tmpl, mapres):
    """Residual vector for fitting a shifted template plus constant offset."""
    yoff, xoff, ampl, meanoff = pars
    centerpix = (np.array(mapunw.shape) - 1) / 2
    y = (np.arange(-centerpix[0], centerpix[0] + 1) - yoff) * mapres
    x = (np.arange(-centerpix[1], centerpix[1] + 1) - xoff) * mapres
    return np.ravel((ampl * tmpl(y, x, grid=True) + meanoff - mapunw) * invnoise)


def gaussfit_source(t_map, q_map, u_map, weight, config=None, band=None, map_res=None):
    """
    Gaussian fitting for T, Q, U maps to determine source amplitudes and Q/U amplitudes.
    Returns T amplitude and position, plus Q and U amplitudes fitted at the T position.

    Parameters:
    -----------
    t_map : array_like
        Temperature map in mK
    q_map : array_like
        Q polarization map in mK
    u_map : array_like
        U polarization map in mK
    weight : object
        Weight map with TT, QQ, UU components
    config : BeamFittingConfig
        Configuration object containing parameters
    band : str, optional
        Identifier of the band associated with the input maps (e.g. \"150GHz\").
        Defaults to the first band listed in the configuration.
    map_res : float, optional
        Pixel resolution in radians. If omitted, ``t_map.res`` is used for
        compatibility with map objects that carry their resolution.

    Returns:
    --------
    tuple
        (yoff_fit, xoff_fit, t_amp, meanoff_fit, q_amp, u_amp)
    """
    map_size_pix = config.map_size_pix
    active_band = band or config.bands[0]
    band_fwhm_arcmin = config.band_fwhm_arcmin[active_band]
    if map_res is None:
        map_res = getattr(t_map, "res", None)
    if map_res is None:
        raise ValueError("gaussfit_source requires map_res when maps do not carry a .res attribute.")

    mapunw = np.asarray(t_map)
    w = np.asarray(weight.TT)
    medwt = np.median(w[w > 0])
    mapunw[w < 0.1 * medwt] = 0.0
    invnoise_t = np.sqrt(weight.TT)
    search_map = np.asarray(t_map) * invnoise_t
    peakpix = np.unravel_index(np.argmax(search_map), search_map.shape)
    peak_guess = mapunw[peakpix[0], peakpix[1]]
    yoff_guess = peakpix[0] - map_size_pix / 2
    xoff_guess = peakpix[1] - map_size_pix / 2
    meanoff_guess = np.average(t_map, weights=w)

    tmpl = make_filtered_gaussian_template(band_fwhm_arcmin * ARC_MIN_RAD / 2.355, map_res)

    pout, _, _, _, ier = optimize.leastsq(
        err_func,
        (yoff_guess, xoff_guess, peak_guess, meanoff_guess),
        args=(mapunw, invnoise_t, tmpl, map_res),
        full_output=True,
    )

    if ier not in [1, 2, 3, 4]:
        raise RuntimeError("T-map Gaussian fit failed.")

    yoff_fit, xoff_fit, t_amp, meanoff_fit = pout

    # Shift offsets from pixel-corner convention (gaussfit output) to pixel centers.
    yoff_fit -= 0.5
    xoff_fit -= 0.5

    # Now fit Q and U amplitudes at the fixed position from T fit
    q_amp = fit_map_amplitude(q_map, weight.QQ, tmpl, yoff_fit, xoff_fit, map_res)
    u_amp = fit_map_amplitude(u_map, weight.UU, tmpl, yoff_fit, xoff_fit, map_res)

    return yoff_fit, xoff_fit, t_amp, meanoff_fit, q_amp, u_amp


def fit_map_amplitude(map_data, weight_data, tmpl, yoff_fixed, xoff_fixed, map_res):
    """
    Fit template amplitude with fixed position.

    Parameters:
    -----------
    map_data : array_like
        Map data to fit
    weight_data : array_like
        Weight data
    tmpl : array_like
        Template for fitting
    yoff_fixed : float
        Fixed y offset
    xoff_fixed : float
        Fixed x offset
    map_res : float
        Pixel resolution in radians

    Returns:
    --------
    float
        Fitted amplitude
    """
    mapunw = np.asarray(map_data)
    w = np.asarray(weight_data)
    medwt = np.median(w[w > 0])
    mapunw[w < 0.1 * medwt] = 0.0
    invnoise = np.sqrt(weight_data)
    meanoff = np.average(map_data, weights=weight_data)

    centerpix = (np.array(map_data.shape) - 1) / 2
    y_pix = max(0, min(int(centerpix[0] + yoff_fixed), map_data.shape[0] - 1))
    x_pix = max(0, min(int(centerpix[1] + xoff_fixed), map_data.shape[1] - 1))
    amp_guess = mapunw[y_pix, x_pix]

    pout, *_ = optimize.leastsq(
        err_func,
        (yoff_fixed, xoff_fixed, amp_guess, meanoff),
        args=(mapunw, invnoise, tmpl, map_res),
        full_output=True,
    )
    return pout[2]
