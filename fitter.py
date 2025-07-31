"""
Polarized beam fitter implementation.

Contains the PolarizedBeamFitter class that handles both ML optimization
and NUTS sampling, supports single and multi-band configurations, and provides
efficient pmap/vmap parallelization across devices.
"""

import jax
import jax.numpy as jnp
import jax.sharding
import numpy as np
import numpyro
import numpyro.distributions as dist
import optimistix as optx
from jax.sharding import Mesh, PartitionSpec
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from spt3g import core, maps

from .beam_model import create_beam_model
from .cache import CacheManager
from .noise_psd import create_noise_psd_calculator
from .source_fitting import gaussfit_source
from .utils import check_zero_fraction, make_apodization_mask, params_from_logit, params_to_logit


def _q16_50_84(x):
    return jnp.quantile(x, jnp.array([0.16, 0.50, 0.84], dtype=jnp.float32), axis=0)


class PolarizedBeamFitter:
    """
    Polarized beam fitting code
    - supports both single and multi-band configurations
    - ML optimization with optimistix or NUTS sampling with numpyro
    - Efficient JAX pmap across devices and vmap within devices giving CPU, single GPU, and multi-GPU support

    Usage example:

    ```python
    from polarized_beam_fitting import PolarizedBeamFitter, BeamFittingConfig
    config = BeamFittingConfig()
    fitter = PolarizedBeamFitter(config=config)
    best_fit_params = fitter.run_fit()
    ```

    Author: Tijmen de Haan <tijmen.dehaan@gmail.com>
    Initial Version: 2025-06-03
    """

    def __init__(self, config=None):
        """
        Initialize the unified polarized beam fitter.

        Parameters:
        -----------
        config : BeamFittingConfig
            Configuration object.
        """
        if config is None:
            raise ValueError("config must be provided")

        # Welcome message
        print(" ")
        print("=================================================================")
        print("==        Welcome to the SPT-3G polarized beam fitter!         ==")
        print("== Questions? Contact Tijmen de Haan <tijmen.dehaan@gmail.com> ==")
        print("=================================================================")
        print(" ")

        self.config = config
        self.bands = config.bands
        self.pol_focus = config.pol_focus
        self.map_shape = (config.map_size_pix, config.map_size_pix)
        self.n_bands = len(self.config.bands)
        self.is_multiband = self.n_bands > 1  # only important for plotting
        print(f"Analysis for {self.bands}")

        # Pre-compute coordinate grids (shared across sources, bands, and stokes parameters)
        print("Pre-computing coordinate grids...")
        ny, nx = self.map_shape
        y_coords = jnp.arange(-ny // 2, ny // 2)
        x_coords = jnp.arange(-nx // 2, nx // 2)
        self.y_grid = jax.device_put(y_coords[:, None] + jnp.zeros(nx, dtype=jnp.float32))
        self.x_grid = jax.device_put(x_coords[None, :] + jnp.zeros((ny, 1), dtype=jnp.float32))

        # Create separate beam models for each band
        self.beam_models = {}
        for band in self.config.bands:
            self.beam_models[band] = create_beam_model(self.config, self.x_grid, self.y_grid)

        self.apod_mask = make_apodization_mask(self.map_shape, config.apodization_width_pix)
        self.apod_mask_jax = jax.device_put(self.apod_mask.astype(jnp.float32))

        # Load and prepare data
        cache_manager = CacheManager(config)
        (
            self.gaussfit_yoff_numpy,
            self.gaussfit_xoff_numpy,
            self.gaussfit_initial_amp_numpy,
            self.raw_maps_numpy,
            self.qu_templates_numpy,
            self.maps_numpy,
            self.maps_fft_numpy,
            self.source_ids,
            self.n_src,
        ) = cache_manager.load_or_create(self._load_and_prepare_data)

        # Pad data for even distribution across devices
        self.n_src_padded, self.source_ids_padded, self.maps_fft_numpy_padded, self.gaussfit_initial_amp_numpy_padded = self._pad_data_for_devices()

        # Initialize noise PSD calculator
        self.noise_psd_calculator = create_noise_psd_calculator(self.config, self.map_shape)

        # For noise calculation, pass the raw maps data
        self.noise_psd_numpy = self.noise_psd_calculator.calculate_noise_psd(self.raw_maps_numpy)

        # Convert to JAX format
        self.psd_dtype = jnp.complex64 if self.config.noise_psd_method == "multiband_covariance" else jnp.float32
        self.noise_psd_jax = jax.device_put(self.noise_psd_numpy.astype(self.psd_dtype))
        self.gaussfit_initial_amp_jax_padded = jax.device_put(self.gaussfit_initial_amp_numpy_padded.astype(jnp.float32))
        self.maps_fft_jax_padded = jax.device_put(self.maps_fft_numpy_padded.astype(jnp.complex64))

        # Initialize parameters using new simple structure
        self.params_physical = self._get_initial_physical_params()
        self.params_logit = params_to_logit(self.params_physical, self.config)

        # Set up JAX Mesh for sharding
        devices = np.array(jax.devices())
        self.mesh = Mesh(devices, ("devices",))

        # Helper specs
        Rep = PartitionSpec()  # replicated
        Sh = PartitionSpec("devices")  # sharded on mesh axis "devices"

        vmap_chi2 = jax.vmap(  # still batch over sources
            self._objective_function_single_source,
            in_axes=(None, 0, 0, 0, 0, 0, None),
        )

        def _chi2_global(beams, yoff, xoff, flux):
            return vmap_chi2(beams, yoff, xoff, flux, self.gaussfit_initial_amp_jax_padded, self.maps_fft_jax_padded, self.noise_psd_jax).sum()

        self._chi2_global = jax.jit(_chi2_global, in_shardings=(Rep, Sh, Sh, Sh), out_shardings=None)

    def _pad_data_for_devices(self):
        """Pad data to be evenly divisible by number of devices."""
        n_devices = jax.local_device_count()
        n_src_real = len(self.source_ids)
        n_src_padded = int(n_devices * np.ceil(n_src_real / n_devices))
        n_pad = n_src_padded - n_src_real

        source_ids_padded = self.source_ids
        maps_fft_numpy_padded = self.maps_fft_numpy
        gaussfit_initial_amp_numpy_padded = self.gaussfit_initial_amp_numpy

        if n_pad > 0:
            print(f"Adding {n_pad} sources to make the new total ({n_src_padded}) divisible by {n_devices} devices.")

            source_ids_padded = np.concatenate([self.source_ids, [f"PAD_{i}" for i in range(n_pad)]])

            ny, nx = self.map_shape
            dummy_shape = (n_pad, ny, nx, self.n_bands, 3)
            dummy_data = np.zeros(dummy_shape, dtype=self.maps_fft_numpy.dtype)
            maps_fft_numpy_padded = np.concatenate([self.maps_fft_numpy, dummy_data], axis=0)

            dummy_amps = np.zeros((n_pad, self.n_bands, 3), dtype=self.gaussfit_initial_amp_numpy.dtype)
            gaussfit_initial_amp_numpy_padded = np.concatenate([self.gaussfit_initial_amp_numpy, dummy_amps], axis=0)

        return n_src_padded, source_ids_padded, maps_fft_numpy_padded, gaussfit_initial_amp_numpy_padded

    def _get_initial_physical_params(self):
        """
        Initialize parameters using the new simple structure.

        Returns:
        --------
        dict
            Physical parameters with structure:
            {
                "beams": [beam_params_band0, beam_params_band1, ...],
                "sources": {
                    "yoff": array (n_src,),
                    "xoff": array (n_src,),
                    "flux_correction": array (n_src, n_bands, 3)
                }
            }
        """
        params_physical = {"beams": [], "sources": {}}

        # Initialize beam parameters for each band
        for band in self.config.bands:
            beam_params = self.beam_models[band].get_initial_physical_params()
            params_physical["beams"].append(beam_params)

        # Initialize source parameters using defaults from config
        yoff_init, xoff_init, flux_init = self.config.source_inits

        params_physical["sources"] = {
            "yoff": jnp.full((self.n_src_padded,), yoff_init, dtype=jnp.float32),
            "xoff": jnp.full((self.n_src_padded,), xoff_init, dtype=jnp.float32),
            "flux_correction": jnp.full((self.n_src_padded, self.n_bands, 3), flux_init, dtype=jnp.float32),
        }

        return params_physical

    def _load_and_prepare_data(self):
        """Load and prepare source data."""
        print("Preparing source data...")
        self._validate_skip_sources()
        gaussfit_yoff, gaussfit_xoff, gaussfit_initial_amp, raw_maps, source_ids, n_src = self._read_coadd_files_and_fit_gaussians()
        normalized_qu_templates = self._create_leakage_template(gaussfit_initial_amp, raw_maps)
        maps_numpy, maps_fft_numpy = self._prepare_fft_maps(gaussfit_initial_amp, raw_maps, normalized_qu_templates)
        return gaussfit_yoff, gaussfit_xoff, gaussfit_initial_amp, raw_maps, normalized_qu_templates, maps_numpy, maps_fft_numpy, source_ids, n_src

    def _validate_skip_sources(self):
        """Validate that skip_sources exist in the data files."""
        all_source_ids = set()
        for coadd_filename in self.config.coadd_filenames:
            g3file = core.G3File(coadd_filename)
            for frame in g3file:
                if frame.type == core.G3FrameType.Map:
                    # Check if any of our target bands are in the frame ID
                    if any(band in frame["Id"] for band in self.bands):
                        all_source_ids.add(frame["Id"])

        missing_skip_sources = [skip for skip in self.config.skip_sources if not any(skip in sid for sid in all_source_ids)]

        if missing_skip_sources:
            print("WARNING: The following skip_sources were not found in any data files:")
            for missing_source in missing_skip_sources:
                print(f"  - {missing_source}")
            print("This may indicate typos in the skip_sources list.")

    def _create_leakage_template(self, gaussfit_initial_amp, raw_maps):
        """Create T->QU leakage correction template."""
        print(f"Creating T->P leakage template using {self.config.leakage_weighting} weighting...")

        if self.config.leakage_weighting == "median":
            normalized_qu_templates = np.median(raw_maps[:, :, :, :, 1:3], axis=0) / gaussfit_initial_amp[:, :, 0]
        else:
            # Proper weighted average over sources with per-source, per-band weights
            # raw_maps shape: (n_src, ny, nx, n_bands, 3)
            # weight_factor shape: (n_src, n_bands)
            # Need to normalize by T amplitude per source per band
            qu_maps = raw_maps[:, :, :, :, 1:3]  # Shape: (n_src, ny, nx, n_bands, 2)
            t_amps = gaussfit_initial_amp[:, :, 0]  # Shape: (n_src, n_bands)

            # Calculate weight factor based on T amplitude
            weight_factor = {"flat": 1.0, "linear": t_amps, "squared": t_amps**2}[self.config.leakage_weighting]  # (n_src, n_bands)

            # Normalize by T amplitude: (n_src, ny, nx, n_bands, 2) / (n_src, 1, 1, n_bands, 1)
            normalized_qu_per_source = qu_maps / t_amps[:, None, None, :, None]

            # Apply weights and average over sources
            # weight_factor shape: (n_src, n_bands) -> (n_src, 1, 1, n_bands, 1)
            weights = weight_factor[:, None, None, :, None]
            weighted_sum = np.sum(normalized_qu_per_source * weights, axis=0)  # Sum over sources
            weight_sum = np.sum(weights, axis=0)  # Sum of weights
            normalized_qu_templates = weighted_sum / weight_sum  # Weighted average

        print(f"Created leakage template from {self.n_src} bright sources")
        return normalized_qu_templates

    def _read_coadd_files_and_fit_gaussians(self):
        """
        Collect data from coadd files, then perform Gaussian fits.

        The reason this is in a single function is that the results of the Gaussian fit decide which sources we
        keep and which we discard.
        """
        gaussfit_results = {}
        for band in self.bands:
            gaussfit_results[band] = {}

        for coadd_filename in self.config.coadd_filenames:
            print(f"Processing file: {coadd_filename}")
            g3file = core.G3File(coadd_filename)
            for frame in g3file:
                if frame.type == core.G3FrameType.Map:
                    source_id = frame["Id"]
                    # Check if this frame corresponds to any of our target bands
                    frame_band = None
                    for band in self.bands:
                        if band in source_id:
                            frame_band = band
                            break

                    if frame_band is None:
                        continue

                    if any(skip in source_id for skip in self.config.skip_sources):
                        continue

                    t_map, q_map, u_map, weight = frame["T"], frame["Q"], frame["U"], frame["Wpol"]
                    maps.remove_weights(t_map, q_map, u_map, weight, zero_nans=False)  # pylint: disable=no-member

                    if not check_zero_fraction(t_map, source_id, max_zero_fraction=self.config.max_zero_fraction):
                        continue

                    yoff_fit, xoff_fit, t_amp_fit, meanoff_fit, q_amp_fit, u_amp_fit = gaussfit_source(t_map, q_map, u_map, weight, config=self.config)

                    if t_amp_fit < self.config.min_t_amplitude:
                        continue

                    # we're later going to multiply q_amp and u_amp, so we better make sure it's not zero
                    pol_amp_min = 0.001  # minimum allowed absolute value
                    q_amp_sign = np.sign(q_amp_fit)
                    u_amp_sign = np.sign(u_amp_fit)
                    q_amp_fit = np.maximum(np.abs(q_amp_fit), pol_amp_min) * q_amp_sign  # overwrite
                    u_amp_fit = np.maximum(np.abs(u_amp_fit), pol_amp_min) * u_amp_sign

                    gaussfit_results[frame_band][source_id] = {
                        "yoff": yoff_fit,
                        "xoff": xoff_fit,
                        "t_amp": t_amp_fit,
                        "meanoff": meanoff_fit,
                        "q_amp": q_amp_fit,
                        "u_amp": u_amp_fit,
                        "maps": (t_map, q_map, u_map, weight),
                    }

        # Extract source IDs, select the ones for which we have all bands, and determine final count
        source_ids_list = list(gaussfit_results[self.bands[0]].keys())
        if len(self.bands) > 1:
            for source_id in source_ids_list.copy():
                for band in self.bands[1:]:
                    if source_id not in gaussfit_results[band]:
                        source_ids_list.remove(source_id)
                        break

        source_ids = np.array(source_ids_list)
        self.source_ids = source_ids
        n_src = len(source_ids)
        self.n_src = n_src
        ny, nx = self.map_shape

        # convert to arrays with the standard ordering (n_source, ny, nx, n_bands, n_stokes)
        gaussfit_yoff = np.zeros((self.n_src, self.n_bands))
        gaussfit_xoff = np.zeros((self.n_src, self.n_bands))
        gaussfit_initial_amp = np.zeros((self.n_src, self.n_bands, 3))
        raw_maps = np.zeros((self.n_src, ny, nx, self.n_bands, 3))

        for band_idx, band in enumerate(self.bands):
            for source_idx, source_id in enumerate(source_ids):
                if source_id in gaussfit_results[band]:
                    result = gaussfit_results[band][source_id]
                    gaussfit_yoff[source_idx, band_idx] = result["yoff"]
                    gaussfit_xoff[source_idx, band_idx] = result["xoff"]
                    gaussfit_initial_amp[source_idx, band_idx, 0] = result["t_amp"]
                    gaussfit_initial_amp[source_idx, band_idx, 1] = result["q_amp"]
                    gaussfit_initial_amp[source_idx, band_idx, 2] = result["u_amp"]
                    raw_maps[source_idx, :, :, band_idx, 0] = result["maps"][0]
                    raw_maps[source_idx, :, :, band_idx, 1] = result["maps"][1]
                    raw_maps[source_idx, :, :, band_idx, 2] = result["maps"][2]

        return gaussfit_yoff, gaussfit_xoff, gaussfit_initial_amp, raw_maps, source_ids, n_src

    def _prepare_fft_maps(self, gaussfit_initial_amp, raw_maps, normalized_qu_templates):
        """Prepare final FFT data using leakage template with consistent array format."""
        # Initialize arrays with consistent ordering: (n_source, ny, nx, n_bands, n_stokes)
        ny, nx = self.map_shape
        map_array = np.zeros((self.n_src, ny, nx, self.n_bands, 3))
        map_fft_array = np.zeros((self.n_src, ny, nx, self.n_bands, 3))

        for source_idx in range(self.n_src):
            for band_idx, band in enumerate(self.bands):
                map_array[source_idx, :, :, band_idx, :] = raw_maps[source_idx, :, :, band_idx, :]

                # Apply T->QU leakage subtraction
                map_array[source_idx, :, :, band_idx, 1] -= normalized_qu_templates[:, :, band_idx, 0] * gaussfit_initial_amp[source_idx, band_idx, 0]
                map_array[source_idx, :, :, band_idx, 2] -= (
                    normalized_qu_templates[:, :, band_idx, 1] * gaussfit_initial_amp[source_idx, band_idx, 0]
                )  # is there a one-line way to do this?

                # Apply apod mask
                map_array_apodized = map_array[source_idx, :, :, band_idx, :] * self.apod_mask[:, :, None]

                # Go to Fourier space
                map_fft_array[source_idx, :, :, band_idx, :] = np.fft.fft2(map_array_apodized, axes=(0, 1))

        print(f"Prepared {self.n_src} sources for fitting.")

        return map_array, map_fft_array

    def _objective_function_single_source(self, beam_params_list, yoff, xoff, flux_correction, initial_amplitudes_source, data_fft_source, noise_psd_source):
        """
        Loss function for a single source, decorated with pmap to run over multiple sources.

        Parameters:
        -----------
        beam_params_list : list
            List of physical beam parameters for each band [beam_params_band0, beam_params_band1, ...]
        yoff : jax.Array
            Y offset of the source. Shape: (n_src,)
        xoff : jax.Array
            X offset of the source. Shape: (n_src,)
        flux_correction : jax.Array
            Flux correction for the source. Shape: (n_src, n_bands, 3)
        initial_amplitudes_source : jax.Array
            Initial T, Q, U amplitudes for the source. Shape: (n_bands, 3)
        data_fft_source : jax.Array
            FFT of the data maps for the source. Shape: (ny, nx, n_bands, 3)
        noise_psd_source : jax.Array
            Noise Power Spectral Density for the source.
            For most PSD types, Shape: (ny, nx, n_bands, 3)
            For the MultiBandCovarianceCalculator, Shape: (ny, nx, n_bands, n_bands, 3, 3)

        Returns:
        --------
        float
            The calculated chi-squared value for the source, normalized by the degrees of freedom.
        """
        # check shapes
        assert len(beam_params_list) == len(self.config.bands), (
            f"_objective_function_single_source received {len(beam_params_list)} beam parameters, expected one for each of {self.config.bands}"
        )
        assert yoff.shape == (), f"_objective_function_single_source received yoff of shape {yoff.shape}, expected a scalar"
        assert xoff.shape == (), f"_objective_function_single_source received xoff of shape {xoff.shape}, expected a scalar"
        assert flux_correction.shape == (self.n_bands, 3), (
            f"_objective_function_single_source received flux_correction of shape {flux_correction.shape}, expected ({self.n_bands}, 3)"
        )
        assert initial_amplitudes_source.shape == (self.n_bands, 3), (
            f"_objective_function_single_source received initial_amplitudes_source of shape {initial_amplitudes_source.shape}, expected ({self.n_bands}, 3)"
        )
        assert data_fft_source.shape == (self.map_shape[0], self.map_shape[1], self.n_bands, 3), (
            f"_objective_function_single_source received data_fft_source of shape {data_fft_source.shape}, expected ({self.map_shape[0]}, {self.map_shape[1]}, {self.n_bands}, 3)"
        )
        assert noise_psd_source.shape == (self.map_shape[0], self.map_shape[1], self.n_bands, 3), (
            f"_objective_function_single_source received noise_psd_source of shape {noise_psd_source.shape}, expected ({self.map_shape[0]}, {self.map_shape[1]}, {self.n_bands}, 3)"
        )

        # Evaluate beam maps for all bands
        beam_maps_T, beam_maps_P = [], []
        for i, band in enumerate(self.config.bands):
            beam_T_map, beam_P_map = self.beam_models[band].evaluate_beam_maps(beam_params_list[i], xoff, yoff)
            beam_maps_T.append(beam_T_map)
            beam_maps_P.append(beam_P_map)

        # Stack beam maps into a single (ny, nx, n_bands, 3) array for T, Q, U
        # The polarization beam map (P) is used for both Q and U components.
        beam_templates = jnp.stack(
            [
                jnp.stack(beam_maps_T, axis=-1),  # T model
                jnp.stack(beam_maps_P, axis=-1),  # Q model
                jnp.stack(beam_maps_P, axis=-1),  # U model
            ],
            axis=-1,
        )

        # Calculate final T, Q, U amplitudes by applying the correction factor
        final_amplitudes = initial_amplitudes_source * flux_correction  # Shape: (n_bands, 3)

        # Create the real-space model via broadcasting: (ny,nx,n_bands,3) * (1,1,n_bands,3)
        model_real = beam_templates * final_amplitudes[None, None, :, :]

        # Apply the apodization mask and transform model to Fourier space
        model_apodized = model_real * self.apod_mask_jax[:, :, None, None]
        model_fft = jnp.fft.fft2(model_apodized, axes=(0, 1))

        # Define polarization weights [T, Q, U]
        pol_weights = jnp.array([1.0, self.pol_focus, self.pol_focus], dtype=jnp.float32)

        # Calculate squared difference between data and model, weighted by noise
        # The result has shape (ny, nx, n_bands, 3)
        chi2_components = jnp.abs(data_fft_source - model_fft) ** 2 / (noise_psd_source + 1e-30)

        # Calculate the mean chi-squared per pixel for each component (band, stokes)
        # Result shape: (n_bands, 3)
        chi2_means_per_component = jnp.mean(chi2_components, axis=(0, 1))

        # Apply polarization focus and sum to get a single chi-squared value
        total_chi2 = jnp.sum(chi2_means_per_component * pol_weights)

        # Normalize by the number of data points (degrees of freedom)
        n_dof = jnp.prod(jnp.array(data_fft_source.shape))

        return total_chi2 / n_dof

    def objective_function(self, params_logit, extra_args=None):
        """
        Total loss function

        Parameters:
        -----------
        params_logit : dict
            Parameters in logit space with structure:
            {"beams": [beam_params_band0, ...], "sources": {"yoff": ..., "xoff": ..., "flux_correction": ...}}
        extra_args : optional
            Unused parameter needed for optimistix compatibility
        """
        params_phys = params_from_logit(params_logit, self.config)

        yoff = params_phys["sources"]["yoff"]
        xoff = params_phys["sources"]["xoff"]
        flux = params_phys["sources"]["flux_correction"]

        total_chi2 = self._chi2_global(
            params_phys["beams"],
            yoff,
            xoff,
            flux,
        )
        return total_chi2  # already replicated, no psum needed

    def _get_individual_chi2s(self, params_logit):
        """Get individual chi2 values for each source (not JIT compiled)."""
        # Convert to physical space
        params_phys = params_from_logit(params_logit, self.config)

        in_axes = (
            None,  # beam_params_list - not sharded (same for all sources)
            0,  # yoff - sharded along axis 0
            0,  # xoff - sharded along axis 0
            0,  # flux_correction - sharded along axis 0
            0,  # initial_amplitudes - sharded along axis 0
            0,  # data_fft_src - sharded along axis 0
            None,  # noise_psd_src - let's assume it's shared (we'll need to fix this for noise PSDs that are more individual)
        )

        def vmap_friendly_loss(beam_params_list, yoff, xoff, flux_correction, initial_amplitudes, data_fft_src, noise_psd_src):
            return self._objective_function_single_source(beam_params_list, yoff, xoff, flux_correction, initial_amplitudes, data_fft_src, noise_psd_src)

        # Use the core objective function
        vmap_loss = jax.vmap(vmap_friendly_loss, in_axes=in_axes)

        all_chi2s = vmap_loss(
            params_phys["beams"],
            params_phys["sources"]["yoff"],
            params_phys["sources"]["xoff"],
            params_phys["sources"]["flux_correction"],
            self.gaussfit_initial_amp_jax_padded,
            self.maps_fft_jax_padded,
            self.noise_psd_jax,
        )
        return all_chi2s

    def run_fit(self):
        """Run ML optimization using optimistix for both single and multi-band configurations."""
        print(f"Starting optimization with max_steps={self.config.n_steps}...")

        solver = optx.BFGS(rtol=1e-12, atol=1e-12)
        y0 = self.params_logit

        with jax.sharding.use_mesh(self.mesh):
            sol = optx.minimise(self.objective_function, solver, y0, max_steps=self.config.n_steps, throw=False)

        if sol.result != optx.RESULTS.successful:
            print(f"ERROR! BFGS did not converge successfully in {self.config.n_steps} steps.")
            print(f"Final physical params: {self.get_physical_params(sol.value)}")
            raise RuntimeError(optx.RESULTS[sol.result])

        self.params_logit = sol.value

        print(f"Optimization finished after {sol.stats['num_steps']} steps.")
        print(optx.RESULTS[sol.result])

        # Calculate final chi2
        final_chi2s = self._get_individual_chi2s(self.params_logit)
        self.latest_chi2s = np.array(final_chi2s)
        print(f"Final chi2: {np.sum(self.latest_chi2s)}")

        return self.get_physical_params(self.params_logit)

    def get_physical_params(self, params_logit=None):
        """Convert parameters from logit space to physical space."""
        if params_logit is None:
            params_logit = self.params_logit
        return params_from_logit(params_logit, self.config)

    # NUTS sampling methods
    def sample_with_nuts_uniform(
        self,
        num_warmup=None,
        num_samples=None,
        num_chains=None,
        chain_method=None,
        max_tree_depth=None,
        target_accept_prob=None,
        dense_mass=None,
        seed=None,
        return_samples=True,
    ):
        """
        Run NUTS sampling with uniform physical priors.
        """
        # Use defaults from config
        num_warmup = self.config.nuts_num_warmup if num_warmup is None else num_warmup
        num_samples = self.config.nuts_num_samples if num_samples is None else num_samples
        num_chains = self.config.nuts_num_chains if num_chains is None else num_chains
        chain_method = self.config.nuts_chain_method if chain_method is None else chain_method
        max_tree_depth = self.config.nuts_max_tree_depth if max_tree_depth is None else max_tree_depth
        target_accept_prob = self.config.nuts_target_accept if target_accept_prob is None else target_accept_prob
        dense_mass = self.config.nuts_dense_mass if dense_mass is None else dense_mass
        seed = self.config.nuts_seed if seed is None else seed

        kernel = NUTS(
            self._nuts_model,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
            dense_mass=dense_mass,
            adapt_step_size=self.config.nuts_adapt_step_size,
            adapt_mass_matrix=self.config.nuts_adapt_mass_matrix,
            find_heuristic_step_size=self.config.nuts_find_heuristic_step_size,
            forward_mode_differentiation=self.config.nuts_forward_mode,
        )

        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            thinning=self.config.nuts_thin,
            progress_bar=self.config.nuts_progress_bar,
        )

        rng_key = jax.random.PRNGKey(seed)
        mcmc.run(rng_key, init_strategy=self._nuts_init())
        flat_samples = mcmc.get_samples(group_by_chain=False)

        # Unflatten samples back into the structured pytree
        samples_phys = self._unflatten_samples(flat_samples)

        # Build summaries
        summary = jax.tree_util.tree_map(lambda x: {"mean": jnp.mean(x, axis=0), "q16_50_84": _q16_50_84(x)}, samples_phys)

        return {
            "samples_phys": samples_phys if return_samples else None,
            "summary": summary,
            "mcmc": mcmc,
        }

    def _unflatten_samples(self, flat_samples):
        """Unflattens the output of mcmc.get_samples() back into our parameter structure."""
        samples_phys = {"beams": [], "sources": {}}

        # Reconstruct beam parameters
        for band_idx, band in enumerate(self.config.bands):
            beam_samples = {}
            for param_name in self.config.beam_coeff_bounds.keys():
                key = f"beam_{band_idx}_{param_name}"
                if key in flat_samples:
                    beam_samples[param_name] = flat_samples[key]
                else:
                    raise KeyError(f"Sample key '{key}' not found in MCMC output.")
            samples_phys["beams"].append(beam_samples)

        # Reconstruct source parameters
        for param_name in ["yoff", "xoff", "flux_correction"]:
            if param_name in flat_samples:
                samples_phys["sources"][param_name] = flat_samples[param_name]
            else:
                raise KeyError(f"Sample key '{param_name}' not found in MCMC output.")

        return samples_phys

    def _nuts_likelihood(self, params_phys):
        """Likelihood function for NUTS that works directly with physical parameters."""
        yoff = params_phys["sources"]["yoff"]
        xoff = params_phys["sources"]["xoff"]
        flux = params_phys["sources"]["flux_correction"]

        # This call assumes _chi2_global can be JITted with these inputs
        total_chi2 = self._chi2_global(
            params_phys["beams"],
            yoff,
            xoff,
            flux,
        )
        return total_chi2

    def _nuts_model(self, *args, **kwargs):
        """Numpyro model with uniform priors in physical space."""
        # Sample all parameters in physical space
        params_phys = self._sample_params_phys_uniform()

        chi2_total = self._nuts_likelihood(params_phys)
        
        norm = jnp.asarray(self.config.chi2_normalization, dtype=jnp.float32)
        numpyro.factor("likelihood", -0.5 * norm * chi2_total)

    def _sample_params_phys_uniform(self):
        """Sample all physical parameters using uniform priors from config bounds."""
        params_phys = {"beams": [], "sources": {}}

        # Sample beam parameters for each band
        for band_idx, band in enumerate(self.config.bands):
            beam_params = {}
            for param_name, bounds in self.config.beam_coeff_bounds.items():
                low, high = jnp.asarray(bounds[0], dtype=jnp.float32), jnp.asarray(bounds[1], dtype=jnp.float32)
                # Create unique name for numpyro
                name = f"beam_{band_idx}_{param_name}"
                # Get shape from initial params
                initial_shape = self.params_physical["beams"][band_idx][param_name].shape
                if initial_shape == ():
                    # Scalar parameter
                    beam_params[param_name] = numpyro.sample(name, dist.Uniform(low, high))
                else:
                    # Array parameter
                    uniform_dist = dist.Uniform(low=jnp.full(initial_shape, low), high=jnp.full(initial_shape, high))
                    beam_params[param_name] = numpyro.sample(name, uniform_dist.to_event(len(initial_shape)))
            params_phys["beams"].append(beam_params)

        # Sample source parameters
        yoff_bounds, xoff_bounds, flux_bounds = self.config.source_bounds

        params_phys["sources"] = {
            "yoff": numpyro.sample("yoff", dist.Uniform(low=jnp.full((self.n_src_padded,), yoff_bounds[0]), high=jnp.full((self.n_src_padded,), yoff_bounds[1])).to_event(1)),
            "xoff": numpyro.sample("xoff", dist.Uniform(low=jnp.full((self.n_src_padded,), xoff_bounds[0]), high=jnp.full((self.n_src_padded,), xoff_bounds[1])).to_event(1)),
            "flux_correction": numpyro.sample(
                "flux_correction",
                dist.Uniform(low=jnp.full((self.n_src_padded, self.n_bands, 3), flux_bounds[0]), high=jnp.full((self.n_src_padded, self.n_bands, 3), flux_bounds[1])).to_event(3),
            ),
        }

        return params_phys

    def _nuts_init(self):
        """Initialize NUTS at current physical parameter values."""
        # Get the current best-fit parameters in physical space
        phys = self.get_physical_params(self.params_logit)

        # Create numpyro-compatible initialization dictionary
        init_vals = {}

        # Add beam parameters
        for band_idx, beam_params in enumerate(phys["beams"]):
            for param_name, param_value in beam_params.items():
                key = f"beam_{band_idx}_{param_name}"
                init_vals[key] = jnp.asarray(param_value, dtype=jnp.float32)

        # Add source parameters
        init_vals["yoff"] = jnp.asarray(phys["sources"]["yoff"], dtype=jnp.float32)
        init_vals["xoff"] = jnp.asarray(phys["sources"]["xoff"], dtype=jnp.float32)
        init_vals["flux_correction"] = jnp.asarray(phys["sources"]["flux_correction"], dtype=jnp.float32)

        return init_to_value(values=init_vals)

    def create_final_model_maps(self, best_fit_params):
        """
        Create final model maps for all sources given the best-fit parameters.
        This is a convenience function for plotting and diagnostics.
        It returns real-space maps, not apodized or FFT'd.
        """
        # Get physical parameters. Assume best_fit_params are physical
        params_phys = best_fit_params

        # Get source and beam parameters for all sources (up to n_src)
        yoffs = np.array(params_phys["sources"]["yoff"][: self.n_src])
        xoffs = np.array(params_phys["sources"]["xoff"][: self.n_src])
        flux_corrections = np.array(params_phys["sources"]["flux_correction"][: self.n_src, :, :])
        beam_params_list = params_phys["beams"]
        initial_amplitudes = self.gaussfit_initial_amp_numpy

        model_maps = {}

        for src_idx in range(self.n_src):
            source_id = self.source_ids[src_idx]
            model_maps[source_id] = {}

            for band_idx, band in enumerate(self.bands):
                # Get source-specific parameters
                yoff = yoffs[src_idx]
                xoff = xoffs[src_idx]

                # Evaluate beam maps using JAX function and convert to numpy
                beam_params = beam_params_list[band_idx]
                beam_T_map_jax, beam_P_map_jax = self.beam_models[band].evaluate_beam_maps(beam_params, yoff, xoff)
                beam_T_map = np.array(beam_T_map_jax)
                beam_P_map = np.array(beam_P_map_jax)

                # Calculate final amplitudes
                final_amplitudes = initial_amplitudes[src_idx, band_idx, :] * flux_corrections[src_idx, band_idx, :]

                # Create real-space model maps for T, Q, U
                model_maps_stokes = {
                    "T": beam_T_map * final_amplitudes[0],
                    "Q": beam_P_map * final_amplitudes[1],
                    "U": beam_P_map * final_amplitudes[2],
                }
                model_maps[source_id][band] = model_maps_stokes

        return model_maps

    def create_beam_profile_maps(self, best_fit_params):
        """Create centered beam maps for radial profile plotting."""
        beam_T_maps = {}
        beam_P_maps = {}
        for band_idx, band in enumerate(self.bands):
            beam_T_map, beam_P_map = self.beam_models[band].evaluate_beam_maps(best_fit_params["beams"][band_idx], 0.0, 0.0)
            beam_T_maps[band] = beam_T_map
            beam_P_maps[band] = beam_P_map
        return beam_T_maps, beam_P_maps
