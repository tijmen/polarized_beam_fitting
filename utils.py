"""
Utility functions for polarized beam fitting.

Contains helper functions for data processing, apodization, and coordinate transformations.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import j0  # pylint: disable=no-name-in-module


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

    # Create smooth hole using cosine taper
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


def compute_2d_asd(map_2d):
    """
    Compute 2D amplitude spectral density from FFT.

    Parameters:
    -----------
    map_2d : array_like
        2D map to analyze

    Returns:
    --------
    asd_2d : array_like
        2D amplitude spectral density (|FFT|)
    """
    # Take FFT and get amplitude
    fft_2d = np.fft.fft2(map_2d)
    asd_2d = np.abs(fft_2d)

    # Shift zero frequency to center
    asd_2d = np.fft.fftshift(asd_2d)

    return asd_2d


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


def linear_interp_differentiable(x, xp, fp):
    """
    1-D linear interpolation that is differentiable w.r.t. x.

    This function provides the same functionality as jnp.interp but maintains
    differentiability with respect to the query points x, which is crucial
    for optimizing position parameters in beam fitting.

    Note: This function may produce NaN gradients at points where the interpolated
    function has zero derivative in all directions (e.g., at the center of symmetric
    functions). The calling code should avoid initializing optimization at such points.

    Parameters:
    -----------
    x : array_like
        Query points to interpolate at
    xp : array_like
        1-D array of x-coordinates, must be increasing
    fp : array_like
        1-D array of function values at xp

    Returns:
    --------
    array_like
        Interpolated values at x
    """
    # Ensure inputs are JAX arrays
    x = jnp.asarray(x, dtype=jnp.float32)
    xp = jnp.asarray(xp, dtype=jnp.float32)
    fp = jnp.asarray(fp, dtype=jnp.float32)

    # Find the indices for interpolation
    # Use continuous indexing and clamp to valid range
    dx = xp[1] - xp[0]  # Assume uniform spacing for simplicity
    idx_float = (x - xp[0]) / dx
    idx = jnp.clip(jnp.floor(idx_float).astype(jnp.int32), 0, xp.size - 2)

    # Get the surrounding points
    x0 = xp[idx]
    x1 = xp[idx + 1]
    y0 = fp[idx]
    y1 = fp[idx + 1]

    # Linear interpolation parameter
    # Add small epsilon to prevent division by zero
    dx_segment = x1 - x0 + 1e-12
    t = jnp.clip((x - x0) / dx_segment, 0.0, 1.0)

    # Linear interpolation
    return y0 + t * (y1 - y0)


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
        B_r[i] = np.trapz(integrand, ell_nonzero) / (2 * np.pi)

    if normalize:
        # Normalize to 1 at r=0
        if B_r[0] != 0:
            B_r = B_r / B_r[0]
            print("  Normalized beam peak to 1.0")
        else:
            print("  Warning: Beam is zero at r=0, cannot normalize")

    print(f"  Beam range after transform: {B_r.min():.6f} to {B_r.max():.6f}")
    return B_r


def load_fieldlevel_data():
    """
    Load field-level beam data from the data files.

    Returns:
    --------
    dict
        Dictionary containing ell and beam data for all bands
    """
    # Get the path to the fieldlevelbeam data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "fieldlevelbeam", "data")

    main_beam_file = os.path.join(data_dir, "B_ell_main_beam.npz")
    rc4_beam_file = os.path.join(data_dir, "B_ell_rc4.npz")

    if not os.path.exists(main_beam_file):
        raise FileNotFoundError(f"Main beam file not found: {main_beam_file}")
    if not os.path.exists(rc4_beam_file):
        raise FileNotFoundError(f"RC4 beam file not found: {rc4_beam_file}")

    print("Loading field-level beam data...")
    print(f"  Main beam: {main_beam_file}")
    print(f"  RC4 beam: {rc4_beam_file}")

    # Load main beam data (Bmain)
    main_data = np.load(main_beam_file)
    ell = main_data["ell"]

    # Load RC4 beam data (BT)
    rc4_data = np.load(rc4_beam_file)

    # Verify ell arrays match
    if not np.array_equal(ell, rc4_data["ell"]):
        raise ValueError("ell arrays don't match between main and RC4 beam files")

    data = {"ell": ell, "bands": ["90", "150", "220"], "Bmain": {}, "BT": {}}

    # Extract beam data for each band
    for band in data["bands"]:
        # Main beam uses format B_ell_{band}
        main_key = f"B_ell_{band}"
        if main_key in main_data:
            data["Bmain"][band] = main_data[main_key]
        else:
            raise KeyError(f"Main beam data for {band} GHz not found (key: {main_key})")

        # RC4 beam uses band as key directly
        if band in rc4_data:
            data["BT"][band] = rc4_data[band]
        else:
            raise KeyError(f"RC4 beam data for {band} GHz not found (key: {band})")

    print(f"Loaded data for bands: {data['bands']}")
    print(f"ell range: {ell.min()} to {ell.max()} ({len(ell)} points)")

    return data


def create_betapol_data():
    """
    Create the betapol.npz file containing real-space beam profiles for all bands.
    """
    print("=" * 60)
    print("Creating betapol.npz from field-level beam data")
    print("=" * 60)

    # Load field-level data
    data = load_fieldlevel_data()
    ell = data["ell"]
    bands = data["bands"]

    # Define radial grid for real-space profiles
    # Use a fine grid from 0 to 10 arcmin
    r_arcmin = np.linspace(0, 10, 1000)
    print(f"\nReal-space grid: {len(r_arcmin)} points from 0 to {r_arcmin.max()} arcmin")

    # Initialize output arrays
    output_data = {"r_fine_arcmin": r_arcmin, "bands": bands}

    # Process each band
    for band in bands:
        print("\n" + "=" * 40)
        print(f"Processing {band} GHz")
        print("=" * 40)

        # Get multipole-space data
        Bmain_ell = data["Bmain"][band]
        BT_ell = data["BT"][band]

        print(f"Bmain_ell range: {Bmain_ell.min():.6f} to {Bmain_ell.max():.6f}")
        print(f"BT_ell range: {BT_ell.min():.6f} to {BT_ell.max():.6f}")

        # Perform Hankel transforms
        print("\nTransforming Bmain...")
        Bmain_r = hankel_transform_beam(ell, Bmain_ell, r_arcmin, normalize=True)

        print("\nTransforming BT...")
        BT_r = hankel_transform_beam(ell, BT_ell, r_arcmin, normalize=True)

        # Store results
        output_data[f"Bmain_r_norm_{band}"] = Bmain_r
        output_data[f"BT_r_norm_{band}"] = BT_r

        print(f"\nCompleted {band} GHz:")
        print(f"  Bmain(r): min={Bmain_r.min():.6f}, max={Bmain_r.max():.6f}")
        print(f"  BT(r): min={BT_r.min():.6f}, max={BT_r.max():.6f}")

    # Save output file
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "betapol.npz")

    print("\n" + "=" * 60)
    print(f"Saving betapol data to: {output_file}")
    np.savez(output_file, **output_data)

    print(f"File saved with keys: {list(output_data.keys())}")
    print(f"File size: {os.path.getsize(output_file) / 1024:.1f} KB")

    return output_file


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


# Parameter transformation utilities
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
    value_scaled = (value - lower) / (upper - lower)
    # Clamp to avoid numerical issues
    value_scaled = jnp.clip(value_scaled, 1e-8, 1.0 - 1e-8)
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
                "flux_correction": array (n_src, n_bands, 3)
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

    # Convert beam parameters for each band
    for _, beam_params in enumerate(physical_params["beams"]):
        beam_logit = {}
        for param_name, param_value in beam_params.items():
            bounds = config.beam_coeff_bounds[param_name]
            beam_logit[param_name] = to_logit(param_value, bounds)
        logit_params["beams"].append(beam_logit)

    # Convert source parameters
    yoff_bounds, xoff_bounds, flux_bounds = config.source_bounds
    logit_params["sources"] = {
        "yoff": to_logit(physical_params["sources"]["yoff"], yoff_bounds),
        "xoff": to_logit(physical_params["sources"]["xoff"], xoff_bounds),
        "flux_correction": to_logit(physical_params["sources"]["flux_correction"], flux_bounds),
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
    for _, beam_logit in enumerate(logit_params["beams"]):
        beam_physical = {}
        for param_name, logit_value in beam_logit.items():
            bounds = config.beam_coeff_bounds[param_name]
            beam_physical[param_name] = from_logit(logit_value, bounds)
        physical_params["beams"].append(beam_physical)

    # Convert source parameters
    yoff_bounds, xoff_bounds, flux_bounds = config.source_bounds
    physical_params["sources"] = {
        "yoff": from_logit(logit_params["sources"]["yoff"], yoff_bounds),
        "xoff": from_logit(logit_params["sources"]["xoff"], xoff_bounds),
        "flux_correction": from_logit(logit_params["sources"]["flux_correction"], flux_bounds),
    }

    return physical_params
