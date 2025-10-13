"""
Source fitting functions for polarized beam analysis.

Contains functions for Gaussian fitting of point sources in T, Q, U maps.

This is mostly copied from the spt3g_software source_fitting module and therefore has the same filename.
"""

import numpy as np
from scipy import optimize
from spt3g import core
from spt3g.sources import fitting as source_fitting


def gaussfit_source(t_map, q_map, u_map, weight, config=None, band=None):
    """
    Gaussian fitting for T, Q, U maps to determine source amplitudes and Q/U amplitudes.
    Returns T amplitude and position, plus Q and U amplitudes fitted at the T position.

    Parameters:
    -----------
    t_map : spt3g map
        Temperature map
    q_map : spt3g map
        Q polarization map
    u_map : spt3g map
        U polarization map
    weight : spt3g weight map
        Weight map with TT, QQ, UU components
    config : BeamFittingConfig
        Configuration object containing parameters
    band : str, optional
        Identifier of the band associated with the input maps (e.g. \"150GHz\").
        Defaults to the first band listed in the configuration.

    Returns:
    --------
    tuple
        (yoff_fit, xoff_fit, t_amp, meanoff_fit, q_amp, u_amp)
    """
    map_size_pix = config.map_size_pix
    active_band = band or config.bands[0]
    band_fwhm_arcmin = config.band_fwhm_arcmin[active_band]
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

    tmpl = source_fitting.make_filtered_gaussian_template(band_fwhm_arcmin * core.G3Units.arcmin / 2.355, t_map.res)

    pout, _, _, _, ier = optimize.leastsq(
        source_fitting.err_func,
        (yoff_guess, xoff_guess, peak_guess, meanoff_guess),
        args=(mapunw, invnoise_t, tmpl, t_map.res),
        full_output=True,
    )

    if ier not in [1, 2, 3, 4]:
        raise RuntimeError("T-map Gaussian fit failed.")

    yoff_fit, xoff_fit, t_amp, meanoff_fit = pout

    # Now fit Q and U amplitudes at the fixed position from T fit
    q_amp = fit_map_amplitude(q_map, weight.QQ, tmpl, yoff_fit, xoff_fit, q_map.res)
    u_amp = fit_map_amplitude(u_map, weight.UU, tmpl, yoff_fit, xoff_fit, u_map.res)

    return yoff_fit, xoff_fit, t_amp, meanoff_fit, q_amp, u_amp


def fit_map_amplitude(map_data, weight_data, tmpl, yoff_fixed, xoff_fixed, map_res):
    """
    Fit template amplitude with fixed position.

    Parameters:
    -----------
    map_data : spt3g map
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
        Map resolution

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
        source_fitting.err_func,
        (yoff_fixed, xoff_fixed, amp_guess, meanoff),
        args=(mapunw, invnoise, tmpl, map_res),
        full_output=True,
    )
    return pout[2]
