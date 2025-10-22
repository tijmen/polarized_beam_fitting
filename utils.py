"""
Utility functions for polarized beam fitting.

Contains helper functions for data processing, apodization, and coordinate transformations.
"""

import re
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates
from scipy.ndimage import shift as nd_shift
from scipy.special import j0  # pylint: disable=no-name-in-module


def parse_declination(source_id):
    """Parses the declination in degrees from a source ID string."""
    # Regex to find negative declination in format DDMM.M
    match = re.search(r"-(\d{2})(\d{2}\.\d)", str(source_id))
    if match:
        deg = float(match.group(1))
        arcmin = float(match.group(2))
        return -(deg + arcmin / 60.0)
    return None  # Return None if pattern not found


def predict_nyquist_ell_x(declination_deg):
    """Predicts the ell_x of the TOD Nyquist frequency feature."""
    if declination_deg is None or np.isnan(declination_deg):
        return np.inf

    scan_speed_az_deg_s = 1.0  # Given scan speed on the bearing
    tod_sampling_hz = 20e6 / 2**17  # ~152.59 Hz
    nyquist_hz = tod_sampling_hz / 2.0

    # Effective scan speed on the sky depends on declination
    on_sky_scan_speed_deg_s = scan_speed_az_deg_s * np.cos(np.deg2rad(declination_deg))

    # Convert temporal frequency (Hz) to spatial frequency (cycles/deg), then to ell
    nyquist_kx_cpd = nyquist_hz / on_sky_scan_speed_deg_s
    return 360.0 * nyquist_kx_cpd


def build_radial_lowpass_mask(ell_radius, ellmax, taper_width_ell):
    """Construct a cosine-tapered radial low-pass mask in ell space."""
    mask = np.ones_like(ell_radius, dtype=float)

    if not np.isfinite(ellmax):
        return mask

    if taper_width_ell <= 0:
        mask = mask.copy()
        mask[ell_radius > ellmax] = 0.0
        return mask

    start = max(ellmax - taper_width_ell, 0.0)
    mask = mask.copy()
    mask[ell_radius > ellmax] = 0.0

    transition = (ell_radius >= start) & (ell_radius <= ellmax)
    if np.any(transition):
        mask[transition] = 0.5 * (1.0 + np.cos(np.pi * (ell_radius[transition] - start) / taper_width_ell))

    return mask


def apply_radial_lowpass(map_2d, apod_mask, ell_radius, ellmax, taper_width_ell):
    """
    Apply a radial low-pass filter with cosine taper in Fourier space.

    Note that this effectively applied the mask, and doesn't unapply it. Therefore,
    this should only be used for building a leakage template.
    """
    if not np.isfinite(ellmax):
        return map_2d

    tapered_map = map_2d * apod_mask
    fourier_map = np.fft.fft2(tapered_map)
    lowpass_mask = build_radial_lowpass_mask(ell_radius, ellmax, taper_width_ell)
    filtered = np.fft.ifft2(fourier_map * lowpass_mask)
    return np.real(filtered)


def ell_grid(map_shape, reso_arcmin):
    """Return ell coordinate grids for the given map geometry."""

    ny, nx = map_shape
    reso_deg = reso_arcmin / 60.0
    ky = np.fft.fftfreq(ny, d=reso_deg)
    kx = np.fft.fftfreq(nx, d=reso_deg)

    # Convert from cycles/degree to ell (ell = 360 * k_cpd)
    ell_y = 360.0 * ky
    ell_x = 360.0 * kx

    ell_y_grid, ell_x_grid = np.meshgrid(ell_y, ell_x, indexing="ij")
    return ell_y, ell_x, ell_y_grid, ell_x_grid


def compute_rectangular_ell_cut_indices(map_shape, reso_arcmin, ellmax):
    """Return ell_y/ell_x indices satisfying |ell_x|, |ell_y| <= ellmax; None if no cutoff."""

    if ellmax is None or not np.isfinite(ellmax) or ellmax <= 0:
        return None, None

    ell_y, ell_x, _, _ = ell_grid(map_shape, reso_arcmin)

    idx_y = np.where(np.abs(ell_y) <= ellmax)[0]
    idx_x = np.where(np.abs(ell_x) <= ellmax)[0]

    assert idx_y.size > 0, f"no indices found below ellmax={ellmax} in y-direction."
    assert idx_x.size > 0, f"no indices found below ellmax={ellmax} in x-direction."

    if idx_y.size == len(ell_y) and idx_x.size == len(ell_x):
        print(f"Warning, asked to cut based on ell<{ellmax}, but no modes found above this value to cut.")
        return None, None

    return idx_y, idx_x


def calculate_tod_nyquist_radial_mask_smooth(source_id, map_shape, config, taper_width_pixels=5):
    """Return a radial low-pass mask based on the TOD Nyquist prediction."""
    ny, nx = map_shape
    reso_deg = config.reso_arcmin / 60.0

    # Get ell coordinates
    ell_y, ell_x, ell_y_grid, ell_x_grid = ell_grid(map_shape, config.reso_arcmin)
    ell_radius = np.sqrt(ell_x_grid**2 + ell_y_grid**2)

    # Compute frequency steps in k-space for taper width conversion
    ky_freq = np.fft.fftfreq(ny, d=reso_deg)
    kx_freq = np.fft.fftfreq(nx, d=reso_deg)
    freq_steps = []
    if ny > 1:
        freq_steps.append(abs(ky_freq[1] - ky_freq[0]))
    if nx > 1:
        freq_steps.append(abs(kx_freq[1] - kx_freq[0]))
    taper_width_cpd = taper_width_pixels * min(freq_steps) if freq_steps else 0.0

    declination = parse_declination(source_id)
    nyquist_ell_x = predict_nyquist_ell_x(declination)
    ellmax = 0.85 * nyquist_ell_x if np.isfinite(nyquist_ell_x) else np.inf
    taper_width_ell = 360.0 * taper_width_cpd

    mask = build_radial_lowpass_mask(ell_radius, ellmax, taper_width_ell)
    return mask.astype(config.dtype_np_real)


def calculate_tod_nyquist_radial_mask(source_id, map_shape, config):
    """
    Same as calculate_tod_nyquist_radial_mask_smooth, but returns a hard
    boolean mask instead of a cosine tapered-mask.
    """
    mask = calculate_tod_nyquist_radial_mask_smooth(source_id, map_shape, config)
    return mask > 0.0


def make_apodization_mask(map_shape, width):
    """
    Creates a 2D cosine apodization mask.

    Parameters:
    -----------
    map_shape : tuple
        Shape of the map (ny, nx)
    width : int
        Width of the apodization region in pixels

    Returns:
    --------
    array_like
        2D apodization mask
    """
    ny, nx = map_shape
    mask = np.ones(map_shape)
    x = np.arange(nx)
    y = np.arange(ny)
    taper = 0.5 * (1 - np.cos(np.pi * np.arange(width) / width))
    mask[y < width, :] *= taper[y[y < width]][:, None]
    mask[y >= ny - width, :] *= taper[ny - 1 - y[y >= ny - width]][:, None]
    mask[:, x < width] *= taper[x[x < width]][None, :]
    mask[:, x >= nx - width] *= taper[nx - 1 - x[x >= nx - width]][None, :]
    return mask


def make_apod_mask_center_excised(map_shape, apod_width, hole_radius_arcmin, reso_arcmin):
    """
    Creates a 2D apodization mask with a smooth hole in the center for noise PSD calculation.

    Parameters:
    -----------
    map_shape : tuple
        Shape of the map (ny, nx)
    apod_width : int
        Width of the edge apodization region in pixels
    hole_radius_arcmin : float
        Radius of the central hole in arcminutes
    reso_arcmin : float
        Map resolution in arcminutes per pixel

    Returns:
    --------
    array_like
        2D apodization mask with central hole
    """
    ny, nx = map_shape

    # Start with regular apodization mask
    mask = make_apodization_mask(map_shape, apod_width)

    # Create coordinate grids and calculate radial distance
    y_coords = np.arange(-ny // 2, ny // 2)
    x_coords = np.arange(-nx // 2, nx // 2)
    y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing="ij")
    r_arcmin = np.sqrt(x_grid**2 + y_grid**2) * reso_arcmin

    # Create smooth hole using cosine taper. If requested hole exceeds the map, return the edge apodization mask.
    max_radius_arcmin = float(np.max(np.sqrt(x_grid**2 + y_grid**2)) * reso_arcmin)
    if hole_radius_arcmin <= 0.0 or hole_radius_arcmin >= max_radius_arcmin:
        return mask

    hole_radius_pix = hole_radius_arcmin / reso_arcmin
    taper_width_pix = hole_radius_pix * 0.2  # 20% of radius for smooth transition

    # Create hole mask: 0 at center, 1 outside hole_radius + taper
    hole_mask = np.ones_like(r_arcmin)

    # Inner region (complete hole)
    inner_boundary = hole_radius_arcmin - taper_width_pix * reso_arcmin
    hole_mask[r_arcmin <= inner_boundary] = 0.0

    # Transition region (smooth taper)
    outer_boundary = hole_radius_arcmin + taper_width_pix * reso_arcmin
    transition_mask = (r_arcmin > inner_boundary) & (r_arcmin < outer_boundary)

    if np.any(transition_mask):
        r_transition = r_arcmin[transition_mask]
        taper_arg = (r_transition - inner_boundary) / (2 * taper_width_pix * reso_arcmin)
        hole_mask[transition_mask] = 0.5 * (1 - np.cos(np.pi * taper_arg))  # cosine taper

    return mask * hole_mask


def check_zero_fraction(t_map, source_id, max_zero_fraction=0.05):
    """
    Check if the source has too many zero pixels and should be skipped.

    Parameters:
    -----------
    t_map : array_like
        Temperature map
    source_id : str
        Source identifier for logging
    max_zero_fraction : float
        Maximum allowed fraction of zero pixels (default: 0.05)

    Returns:
    --------
    bool
        True if source should be kept, False if it should be skipped
    """
    t_array = np.asarray(t_map)
    zero_pixels = np.sum(t_array == 0)
    total_pixels = t_array.size
    zero_fraction = zero_pixels / total_pixels

    if zero_fraction > max_zero_fraction:
        print(f"Skipping source {source_id} because it has {zero_pixels}/{total_pixels} ({zero_fraction:.3f}) zero pixels")
        return False
    return True


def shift_map_bilinear(map_2d, y_shift, x_shift, mode="nearest"):
    """Shift a 2D map using bilinear interpolation."""
    return nd_shift(map_2d, shift=(y_shift, x_shift), order=1, mode=mode, prefilter=False)


def safe_filename(source_id):
    """
    Convert source ID to a safe filename by replacing problematic characters.

    Parameters:
    -----------
    source_id : str
        Source identifier

    Returns:
    --------
    str
        Safe filename string
    """
    # Replace problematic characters with underscores
    safe_name = source_id.replace(".", "_").replace("-", "_").replace("+", "_").replace(" ", "_")
    return safe_name


def linear_interp_differentiable(x, xp, fp, config):
    """
    1-D linear interpolation that is differentiable w.r.t. x.

    Uses jax.scipy.ndimage.map_coordinates for a fully differentiable
    interpolation, replacing the previous manual implementation that had
    discontinuities at integer boundaries.

    Parameters:
    -----------
    x : array_like
        Query points to interpolate at.
    xp : array_like
        1-D array of x-coordinates, must be uniformly spaced and increasing.
    fp : array_like
        1-D array of function values at xp.
    config : BeamFittingConfig
        Configuration object.

    Returns:
    --------
    array_like
        Interpolated values at x.
    """
    x = jnp.asarray(x, dtype=config.dtype_jax_real)
    xp = jnp.asarray(xp, dtype=config.dtype_jax_real)
    fp = jnp.asarray(fp, dtype=config.dtype_jax_real)

    # Convert physical coordinates (x) to fractional index coordinates
    # for map_coordinates. Assumes a uniform grid for xp.
    dx = xp[1] - xp[0]
    coords_frac = (x - xp[0]) / dx

    # map_coordinates expects coordinates in a specific shape: (n_dims, ...).
    # Since our interpolation is 1D, we reshape to (1, ...).
    coords_reshaped = coords_frac.reshape(1, *coords_frac.shape)

    # Perform the differentiable interpolation. order=1 specifies linear.
    # mode='nearest' handles boundary conditions by extending the edge value.
    interpolated_values = map_coordinates(fp, coords_reshaped, order=1, mode="nearest")

    return interpolated_values


def hankel_transform_beam(ell, B_ell, r_arcmin, normalize=True):
    """
    Perform Hankel transform to convert from multipole space to real space.

    For a circularly symmetric beam, the relationship is:
    B(r) = (1/2π) * ∫ B_ell * J_0(ell * r * π/180/60) * ell * d_ell

    where r is in arcminutes and we convert to radians for the Bessel function.

    Parameters:
    -----------
    ell : array
        Multipole values
    B_ell : array
        Beam in multipole space
    r_arcmin : array
        Radial coordinates in arcminutes
    normalize : bool
        Whether to normalize the beam to 1 at r=0

    Returns:
    --------
    array
        Beam profile in real space
    """
    print("Performing Hankel transform...")
    print(f"  ell range: {ell.min()} to {ell.max()}")
    print(f"  r range: {r_arcmin.min():.3f} to {r_arcmin.max():.3f} arcmin")

    # Convert arcminutes to radians for Bessel function
    r_rad = r_arcmin * np.pi / (180 * 60)

    # Initialize output array
    B_r = np.zeros_like(r_arcmin)

    # Perform the integral using trapezoidal rule
    # Skip ell=0 to avoid issues with normalization
    mask = ell > 0
    ell_nonzero = ell[mask]
    B_ell_nonzero = B_ell[mask]

    for i, r in enumerate(r_rad):
        if i % 50 == 0:
            print(f"  Processing r index {i}/{len(r_rad)}")

        # J_0(ell * r) for this radius
        bessel_values = j0(ell_nonzero * r)

        # Integrand: B_ell * J_0(ell * r) * ell
        integrand = B_ell_nonzero * bessel_values * ell_nonzero

        # Integrate using trapezoidal rule
        # Factor of 1/(2π) from the Hankel transform definition
        B_r[i] = np.trapezoid(integrand, ell_nonzero) / (2 * np.pi)

    if normalize:
        # Normalize to 1 at r=0
        if B_r[0] != 0:
            B_r = B_r / B_r[0]
            print("  Normalized beam peak to 1.0")
        else:
            print("  Warning: Beam is zero at r=0, cannot normalize")

    print(f"  Beam range after transform: {B_r.min():.6f} to {B_r.max():.6f}")
    return B_r


def get_stokes_name(index):
    """
    Get the Stokes parameter name from index.

    Parameters:
    -----------
    index : int
        Stokes index (0, 1, or 2)

    Returns:
    --------
    str
        Stokes parameter name: 0="T", 1="Q", 2="U"
    """
    stokes_names = ["T", "Q", "U"]
    return stokes_names[index]


def convert_dict_to_array(data_dict, bands, map_shape=None):
    """
    Convert dictionary-based data to array format (y,x,band,stokes).

    Parameters:
    -----------
    data_dict : dict
        Dictionary with structure {band: {stokes: data}}
    bands : list
        List of band names
    map_shape : tuple, optional
        Shape (ny, nx) for map data. If None, infers from first data entry.

    Returns:
    --------
    np.ndarray
        Array with shape (ny, nx, n_bands, 3) for map data or
        (n_bands, 3) for scalar data
    """
    n_bands = len(bands)

    # Get a sample data entry to determine shape
    sample_band = bands[0]
    sample_data = data_dict[sample_band]["T"]

    if map_shape is None and hasattr(sample_data, "shape"):
        if len(sample_data.shape) == 2:
            map_shape = sample_data.shape
        else:
            map_shape = None

    # Create output array
    if map_shape is not None:
        # Map-like data: (y, x, band, stokes)
        ny, nx = map_shape
        output_array = np.zeros((ny, nx, n_bands, 3), dtype=sample_data.dtype)

        for band_idx, band in enumerate(bands):
            for stokes_idx, stokes in enumerate(["T", "Q", "U"]):
                output_array[:, :, band_idx, stokes_idx] = data_dict[band][stokes]
    else:
        # Scalar-like data: (band, stokes)
        output_array = np.zeros((n_bands, 3), dtype=type(sample_data))

        for band_idx, band in enumerate(bands):
            for stokes_idx, stokes in enumerate(["T", "Q", "U"]):
                output_array[band_idx, stokes_idx] = data_dict[band][stokes]

    return output_array


def convert_array_to_dict(data_array, bands):
    """
    Convert array format back to dictionary format.

    Parameters:
    -----------
    data_array : np.ndarray
        Array with shape (ny, nx, n_bands, 3) or (n_bands, 3)
    bands : list
        List of band names

    Returns:
    --------
    dict
        Dictionary with structure {band: {stokes: data}}
    """
    data_dict = {}

    for band_idx, band in enumerate(bands):
        data_dict[band] = {}
        for stokes_idx, stokes in enumerate(["T", "Q", "U"]):
            if len(data_array.shape) == 4:
                # Map-like data
                data_dict[band][stokes] = data_array[:, :, band_idx, stokes_idx]
            else:
                # Scalar-like data
                data_dict[band][stokes] = data_array[band_idx, stokes_idx]

    return data_dict


def to_logit(value, bounds):
    """
    Transform physical parameter to logit space.

    Parameters:
    -----------
    value : float or array
        Physical parameter value
    bounds : tuple
        (lower, upper) bounds for the parameter

    Returns:
    --------
    float or array
        Parameter in logit space
    """
    lower, upper = bounds
    lower = jnp.broadcast_to(lower, value.shape)  # check if this makes a difference, maybe for jax.grad?
    upper = jnp.broadcast_to(upper, value.shape)
    value_scaled = (value - lower) / (upper - lower)
    # Clamp to avoid numerical issues
    value_scaled = jnp.clip(value_scaled, 1e-18, 1.0 - 1e-18)
    return jnp.log(value_scaled / (1.0 - value_scaled))


def from_logit(logit_value, bounds):
    """
    Transform logit parameter to physical space.

    Parameters:
    -----------
    logit_value : float or array
        Parameter in logit space
    bounds : tuple
        (lower, upper) bounds for the parameter

    Returns:
    --------
    float or array
        Physical parameter value
    """
    lower, upper = bounds
    lower = jnp.broadcast_to(lower, logit_value.shape)
    upper = jnp.broadcast_to(upper, logit_value.shape)
    return lower + (upper - lower) * jax.nn.sigmoid(logit_value)


def params_to_logit(physical_params, config):
    """
    Convert entire physical parameter dictionary to logit space for optimization.

    Parameters:
    -----------
    physical_params : dict
        Physical parameters with structure:
        {
            "beams": [beam_params_band0, beam_params_band1, ...],
            "sources": {
                "yoff": array (n_src,),
                "xoff": array (n_src,),
                "flux": array (n_src, n_bands, 3)
            }
        }
    config : BeamFittingConfig
        Configuration object with bounds

    Returns:
    --------
    dict
        Parameters in logit space with same structure
    """
    logit_params = {"beams": [], "sources": {}}
    active_model_bounds = config.active_beam_model_bounds

    # Convert beam parameters for each band
    for _, beam_params in enumerate(physical_params["beams"]):
        beam_logit = {}
        for param_name, param_value in beam_params.items():
            bounds = active_model_bounds[param_name]
            beam_logit[param_name] = to_logit(param_value, bounds)
        logit_params["beams"].append(beam_logit)

    # Convert source parameters
    yoff_bounds, xoff_bounds = config.source_position_bounds
    flux_bounds = config.source_flux_bounds
    logit_params["sources"] = {
        "yoff": to_logit(physical_params["sources"]["yoff"], yoff_bounds),
        "xoff": to_logit(physical_params["sources"]["xoff"], xoff_bounds),
        "flux": to_logit(physical_params["sources"]["flux"], flux_bounds),
    }

    return logit_params


def params_from_logit(logit_params, config):
    """
    Convert logit parameter dictionary back to physical space.

    Parameters:
    -----------
    logit_params : dict
        Parameters in logit space
    config : BeamFittingConfig
        Configuration object with bounds

    Returns:
    --------
    dict
        Physical parameters
    """
    physical_params = {"beams": [], "sources": {}}

    # Convert beam parameters for each band
    active_model_bounds = config.active_beam_model_bounds

    for _, beam_logit in enumerate(logit_params["beams"]):
        beam_physical = {}
        for param_name, logit_value in beam_logit.items():
            bounds = active_model_bounds[param_name]
            beam_physical[param_name] = from_logit(logit_value, bounds)
        physical_params["beams"].append(beam_physical)

    # Convert source parameters
    yoff_bounds, xoff_bounds = config.source_position_bounds
    flux_bounds = config.source_flux_bounds
    physical_params["sources"] = {
        "yoff": from_logit(logit_params["sources"]["yoff"], yoff_bounds),
        "xoff": from_logit(logit_params["sources"]["xoff"], xoff_bounds),
        "flux": from_logit(logit_params["sources"]["flux"], flux_bounds),
    }

    return physical_params


# Whitening transform utilities for NUTS
def build_whitening_transform(map_params, curvature):
    """
    Builds functions to transform between physical and whitened parameter spaces.

    Args:
        map_params (pytree): The parameters at the maximum a posteriori (MAP) estimate.
        curvature (pytree): The diagonal of the Hessian (second derivatives) at the MAP.

    Returns:
        (to_whitened, from_whitened): A tuple of transformation functions.
    """
    # Flatten the pytrees into 1D vectors and get the unflattening function
    map_vector, unflatten_fn = jax.flatten_util.ravel_pytree(map_params)
    curvature_vector, _ = jax.flatten_util.ravel_pytree(curvature)

    # The scale is the sqrt of the curvature (diagonal of Hessian)
    # Add a small epsilon to avoid division by zero or taking sqrt of negative numbers
    scale_vector = jnp.sqrt(jnp.maximum(curvature_vector, 1e-12))

    def to_whitened(params_phys):
        params_vector, _ = jax.flatten_util.ravel_pytree(params_phys)
        return (params_vector - map_vector) * scale_vector

    def from_whitened(params_white):
        # params_white is already a flat vector
        params_vector = (params_white / scale_vector) + map_vector
        return unflatten_fn(params_vector)

    # The log-determinant of the Jacobian of the `from_whitened` transform.
    # This is a constant correction term for the potential energy in NUTS.
    # det(J) = det(diag(1/scale)) = 1 / product(scale_vector)
    # log|det(J)| = -sum(log(scale_vector))
    log_det_jacobian = -jnp.sum(jnp.log(scale_vector))

    return to_whitened, from_whitened, log_det_jacobian


def _clone_params(params):
    """Create a tree copy of parameters stored as the current best."""
    if params is None:
        return None
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)


def init_convergence_state() -> Dict[str, Any]:
    """Return a fresh convergence tracking state."""
    return {"loss_history": (), "best_loss": float("inf"), "best_step": -1, "best_params": None}


# Convergence criteria utilities
def check_convergence(
    loss,
    grad_norm,
    step,
    config,
    convergence_state: Optional[Dict[str, Any]] = None,
    initial_grad_norm=None,
    current_params=None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Check if convergence criterion is met and optionally track best parameters.

    Parameters:
    -----------
    loss : float
        Current loss value
    grad_norm : float
        Current gradient norm
    step : int
        Current optimization step
    config : BeamFittingConfig
        Configuration object containing convergence parameters
    convergence_state : dict, optional
        State snapshot for tracking convergence (a new instance is returned)
    initial_grad_norm : float, optional
        Initial gradient norm (required for relative_gtol)
    current_params : pytree, optional
        Current parameters to track alongside best loss

    Returns:
    --------
    tuple
        (converged: bool, message: str, state: dict)
    """
    # Initialize convergence state if not provided
    if convergence_state is None:
        state = init_convergence_state()
    else:
        state = dict(convergence_state)

    criterion_type = config.convergence_criterion

    if criterion_type == "absolute_gtol":
        converged = grad_norm < config.absolute_gtol
        message = f"gradient norm {grad_norm:.2e} < {config.absolute_gtol:.2e}" if converged else ""
        return converged, message, state

    elif criterion_type == "relative_gtol":
        if initial_grad_norm is None:
            raise ValueError("relative_gtol requires initial_grad_norm")
        relative_grad = grad_norm / initial_grad_norm
        converged = relative_grad < config.relative_gtol
        message = f"relative gradient {relative_grad:.2e} < {config.relative_gtol:.2e}" if converged else ""
        return converged, message, state

    elif criterion_type == "loss_history":
        # Update best loss and params if current loss is better
        if loss < state["best_loss"]:
            state["best_loss"] = float(loss)
            state["best_step"] = step
            state["best_params"] = _clone_params(current_params)

        # Add current loss to history with rolling window
        history_length = config.loss_history_length
        history = state["loss_history"] + (float(loss),)
        if len(history) > history_length:
            history = history[-history_length:]
        state["loss_history"] = history

        # Check if we have enough history and no improvement
        if len(state["loss_history"]) >= history_length:
            steps_since_best = step - state["best_step"]
            converged = steps_since_best >= history_length
            message = f"no improvement for {steps_since_best} steps" if converged else ""
            return converged, message, state

        # Not enough history yet
        return False, "", state

    else:
        raise ValueError(f"Unknown convergence criterion: {criterion_type}")


# Newton-PCG utilities
def hvp(loss_fn, params, v):
    """
    Hessian-vector product using forward-mode AD (Pearlmutter trick).

    Parameters:
    -----------
    loss_fn : callable
        Loss function that takes params and returns scalar
    params : pytree
        Parameters at which to evaluate HVP
    v : pytree
        Vector to multiply with Hessian

    Returns:
    --------
    pytree
        Hessian-vector product H @ v
    """
    g = jax.grad(loss_fn)
    _, hv = jax.jvp(g, (params,), (v,))
    return hv


def pcg_solve(Ax, b, M_inv=None, tol=1e-12, maxiter=1000):
    """
    Preconditioned conjugate gradient solver for Ax = b.

    Parameters:
    -----------
    Ax : callable
        Function that applies matrix A to a vector
    b : array
        Right-hand side vector
    M_inv : callable, optional
        Preconditioner (applies M^-1 to a vector)
    tol : float
        Convergence tolerance
    maxiter : int
        Maximum iterations

    Returns:
    --------
    tuple
        (solution, residual_norm)
    """
    x = jnp.zeros_like(b)
    r = b - Ax(x)
    z = r if M_inv is None else M_inv(r)
    p = z
    rz_old = jnp.dot(r, z)
    b_norm = jnp.linalg.norm(b)

    def cond(state):
        k, x, r, z, p, rz_old = state
        return jnp.logical_and(k < maxiter, jnp.linalg.norm(r) > tol * b_norm)

    def body(state):
        k, x, r, z, p, rz_old = state
        Ap = Ax(p)
        alpha = rz_old / jnp.dot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        z_new = r_new if M_inv is None else M_inv(r_new)
        rz_new = jnp.dot(r_new, z_new)
        beta = rz_new / rz_old
        p_new = z_new + beta * p
        return (k + 1, x_new, r_new, z_new, p_new, rz_new)

    k0 = jnp.array(0)
    state = (k0, x, r, z, p, rz_old)
    state = jax.lax.while_loop(cond, body, state)
    _, x_final, r_final, *_ = state
    return x_final, jnp.linalg.norm(r_final)


def newton_step_pcg_whitened(loss_fn_white, params_white, damping=1e-6, cg_tol=1e-12, cg_maxiter=1000):
    """
    Single Newton step in whitened space using preconditioned conjugate gradient.
    In whitened space, the Hessian is approximately identity, so we only need damping.

    Parameters:
    -----------
    loss_fn_white : callable
        Loss function in whitened space (takes whitened params, returns scalar)
    params_white : array
        Current parameters in whitened space (flat array)
    damping : float
        Damping term λ for regularization
    cg_tol : float
        CG convergence tolerance
    cg_maxiter : int
        Maximum CG iterations

    Returns:
    --------
    tuple
        (new_params_white, cg_residual_norm, gradient_norm)
    """
    # Compute gradient in whitened space
    g_white = jax.grad(loss_fn_white)(params_white)
    grad_norm = jnp.linalg.norm(g_white)

    # Define Hessian-vector product operator with damping
    def hvp_white(v):
        hv = hvp(loss_fn_white, params_white, v)
        return hv + damping * v

    # In whitened space, Hessian ≈ I, so no preconditioner needed
    b = -g_white
    s_white, cg_res = pcg_solve(hvp_white, b, M_inv=None, tol=cg_tol, maxiter=cg_maxiter)

    # Update parameters in whitened space
    params_white_new = params_white + s_white

    return params_white_new, cg_res, grad_norm
