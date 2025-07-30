"""
Polarized beam fitter implementation.

Contains the PolarizedBeamFitter class that handles both ML optimization
and NUTS sampling, supports single and multi-band configurations, and provides
efficient pmap/vmap parallelization across devices.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optimistix as optx
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from spt3g import core, maps
from .utils import make_apodization_mask, check_zero_fraction
from .source_fitting import gaussfit_source
from .beam_model import create_beam_model
from .noise_psd import create_noise_psd_calculator
from .cache import CacheManager
from .param_manager import ParameterManager


def _q16_50_84(x):
    return jnp.quantile(x, jnp.array([0.16, 0.50, 0.84], dtype=jnp.float32), axis=0)


class PolarizedBeamFitter:
    """
    Polarized beam fitter supporting both single and multi-band configurations.
    - ML optimization with optimistix or NUTS sampling with numpyro
    - Efficient JAX pmap across devices and vmap within devices
    - CPU, single GPU, and multi-GPU support

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
        self.n_src_padded, self.source_ids_padded, self.maps_fft_numpy_padded, self.gaussfit_initial_amp_numpy_padded, self.gaussfit_yoff_numpy_padded, self.gaussfit_xoff_numpy_padded = self._pad_data_for_devices()

        # Initialize noise PSD calculator
        self.noise_psd_calculator = create_noise_psd_calculator(self.config, self.map_shape)

        # For noise calculation, pass the raw maps data
        self.noise_psd_numpy = self.noise_psd_calculator.calculate_noise_psd(self.raw_maps_numpy)

        # Convert to JAX format
        self.psd_dtype = jnp.complex64 if self.config.noise_psd_method == "multiband_covariance" else jnp.float32
        self.noise_psd_jax = jax.device_put(self.noise_psd_numpy.astype(self.psd_dtype))
        self.gaussfit_initial_amp_jax_padded = jax.device_put(self.gaussfit_initial_amp_numpy_padded.astype(jnp.float32))
        self.maps_fft_jax_padded = jax.device_put(self.maps_fft_numpy_padded.astype(jnp.complex64))
        self.gaussfit_yoff_jax_padded = jax.device_put(self.gaussfit_yoff_numpy_padded.astype(jnp.float32))
        self.gaussfit_xoff_jax_padded = jax.device_put(self.gaussfit_xoff_numpy_padded.astype(jnp.float32))

        # Initialize parameter manager and parameters
        self.param_manager = ParameterManager(self.config, self.beam_models, self.n_src_padded)
        self.params = self.param_manager.get_initial_params(
            initial_yoff=self.gaussfit_yoff_jax_padded,
            initial_xoff=self.gaussfit_xoff_jax_padded,
            in_logit_space=True
        )
        self.initial_params_physical = self.param_manager.to_physical(self.params)
        self._pmapped_chi2_calculator = self._create_pmapped_objective()

    def _pad_data_for_devices(self):
        """Pad data to be evenly divisible by number of devices."""
        n_devices = jax.local_device_count()
        n_src_real = len(self.source_ids)
        n_src_padded = int(n_devices * np.ceil(n_src_real / n_devices))  # round up to the nearest multiple of n_devices
        n_pad = n_src_padded - n_src_real

        # Start with unpadded arrays
        source_ids_padded = self.source_ids
        maps_fft_numpy_padded = self.maps_fft_numpy
        gaussfit_initial_amp_numpy_padded = self.gaussfit_initial_amp_numpy
        # THE FIX: Initialize padded yoff/xoff with the unpadded versions
        gaussfit_yoff_numpy_padded = self.gaussfit_yoff_numpy
        gaussfit_xoff_numpy_padded = self.gaussfit_xoff_numpy

        if n_pad > 0:
            print(f"Adding {n_pad} sources to make the new total ({n_src_padded}) divisible by {n_devices} devices.")

            # Pad source_ids
            source_ids_padded = np.concatenate([self.source_ids, [f"PAD_{i}" for i in range(n_pad)]])

            # Pad data array
            ny, nx = self.map_shape
            dummy_shape = (n_pad, ny, nx, self.n_bands, 3)
            dummy_data = np.zeros(dummy_shape, dtype=self.maps_fft_numpy.dtype)
            maps_fft_numpy_padded = np.concatenate([self.maps_fft_numpy, dummy_data], axis=0)

            # Pad initial amplitudes
            dummy_amps = np.zeros((n_pad, self.n_bands, 3), dtype=self.gaussfit_initial_amp_numpy.dtype)
            gaussfit_initial_amp_numpy_padded = np.concatenate([self.gaussfit_initial_amp_numpy, dummy_amps], axis=0)
            
            dummy_offsets = np.zeros((n_pad, self.n_bands), dtype=self.gaussfit_yoff_numpy.dtype)
            gaussfit_yoff_numpy_padded = np.concatenate([self.gaussfit_yoff_numpy, dummy_offsets], axis=0)
            gaussfit_xoff_numpy_padded = np.concatenate([self.gaussfit_xoff_numpy, dummy_offsets], axis=0)

        # Return all the new padded quantities
        return n_src_padded, source_ids_padded, maps_fft_numpy_padded, gaussfit_initial_amp_numpy_padded, gaussfit_yoff_numpy_padded, gaussfit_xoff_numpy_padded

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
                    maps.remove_weights(t_map, q_map, u_map, weight, zero_nans=False)  # IGNORE pylint errors, this is correct

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
        gaussfit_amplitudes = np.zeros((self.n_src, self.n_bands, 3))

        for source_idx in range(self.n_src):
            for band_idx, band in enumerate(self.bands):
                map_array[source_idx, :, :, band_idx, :] = raw_maps[source_idx, :, :, band_idx, :]

                # Apply T->QU leakage subtraction
                map_array[source_idx, :, :, band_idx, 1] -= normalized_qu_templates[:, :, band_idx, 0] * gaussfit_initial_amp[source_idx, band_idx, 0]
                map_array[source_idx, :, :, band_idx, 2] -= (
                    normalized_qu_templates[:, :, band_idx, 1] * gaussfit_initial_amp[source_idx, band_idx, 0]
                )  # is there a one-line way to do this?

                # Apply apod mask
                map_array_apodized = map_array[source_idx, :, :, band_idx, :] * self.apod_mask.reshape(ny, nx, 1)

                # Go to Fourier space
                map_fft_array[source_idx, :, :, band_idx, :] = np.fft.fft2(map_array_apodized, axes=(0, 1))

        print(f"Prepared {self.n_src} sources for fitting.")

        return map_array, map_fft_array

    def _objective_function_single_source_core(self, beam_params_phys, source_params_phys, initial_amplitudes_source, data_fft_source, noise_psd_source):
        """
        Loss function for a single source, vectorized across bands and Stokes parameters.
        Now takes physical parameters directly (no logit conversion needed).

        Parameters:
        -----------
        beam_params_phys : dict
            Dictionary of physical beam parameters for all bands {band: {param: value}}
        source_params_phys : dict
            Dictionary of physical source parameters ('yoff', 'xoff', 'flux_correction')
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
        # Get physical source position and flux correction factors (already in physical space)
        dy = source_params_phys["yoff"]
        dx = source_params_phys["xoff"]
        flux_correction_factors = source_params_phys["flux_correction"]  # Shape: (n_bands, 3)

        # Evaluate beam maps for all bands
        beam_maps_T, beam_maps_P = [], []
        for i, band in enumerate(self.config.bands):
            beam_T_map, beam_P_map = self.beam_models[band].evaluate_beam_maps(beam_params_phys[band], dx, dy)
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
        final_amplitudes = initial_amplitudes_source * flux_correction_factors  # Shape: (n_bands, 3)

        # Create the real-space model via broadcasting: (ny,nx,n_bands,3) * (1,1,n_bands,3)
        model_real = beam_templates * final_amplitudes[None, None, :, :]

        # Apply the apodization mask and transform model to Fourier space
        model_apodized = model_real * self.apod_mask_jax[:, :, None, None]
        model_fft = jnp.fft.fft2(model_apodized, axes=(0, 1))

        # --- Chi-squared Calculation in Fourier Space ---

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

#  Here's the proper version:
    # def _create_pmapped_objective(self):
    #     """
    #     Creates and JIT-compiles the pmapped objective function.
    #     """

    #     def _pmapped_chi2_calculator(beam_params_phys, source_params_phys, initial_amplitudes, data_fft, noise_psd):
    #         # We will vmap the core objective function directly.
    #         vmapped_core_function = jax.vmap(
    #             self._objective_function_single_source_core,
    #             in_axes=(
    #                 None,                                        # beam_params_phys
    #                 {'yoff': 0, 'xoff': 0, 'flux_correction': 0}, # source_params_phys
    #                 0,                                           # initial_amplitudes_source
    #                 0,                                           # data_fft_source
    #                 None,                                        # noise_psd_source
    #             )
    #         )

    #         # Call the vmapped function with the full (per-device) arrays.
    #         # This will now work because all mapped arrays have the same leading dimension.
    #         chi2_values = vmapped_core_function(
    #             beam_params_phys,
    #             source_params_phys,
    #             initial_amplitudes,
    #             data_fft,
    #             noise_psd
    #         )

    #         return jnp.sum(chi2_values)

    #     pmapped_calculator = jax.pmap(
    #         _pmapped_chi2_calculator,
    #         in_axes=(
    #             None,
    #             {"yoff": 0, "xoff": 0, "flux_correction": 0},
    #             0,
    #             0,
    #             None,
    #         ),
    #         axis_name="devices",
    #     )
    #     return pmapped_calculator

# and this is the shitty fori_loop version for debugging
    def _create_pmapped_objective(self):
        """
        Creates and JIT-compiles the pmapped objective function using a fori_loop for robust iteration.
        """

        def _pmapped_chi2_calculator(beam_params_phys, source_params_phys, initial_amplitudes, data_fft, noise_psd):
            # This function operates on a slice of data for each device.
            n_src_per_device = data_fft.shape[0]

            # Define the body of the loop. It takes an index `i` and the loop state (the running sum `chi2_sum`).
            def loop_body(i, chi2_sum):
                # Manually slice the pytrees and arrays for the i-th source.
                # jax.tree_util.tree_map is used to slice each leaf of the source_params_phys pytree.
                source_p_slice = jax.tree_util.tree_map(lambda x: x[i], source_params_phys)
                initial_a_slice = initial_amplitudes[i]
                data_f_slice = data_fft[i]

                # Call the core function on the single-source slices.
                # Note that beam_params_phys and noise_psd are captured from the outer scope and are not sliced.
                chi2_single = self._objective_function_single_source_core(
                    beam_params_phys,
                    source_p_slice,
                    initial_a_slice,
                    data_f_slice,
                    noise_psd,
                )
                # Add the result to the running sum.
                return chi2_sum + chi2_single

            # Initialize the loop with a starting value of 0.0 and run it from i=0 to n_src_per_device-1.
            total_chi2 = jax.lax.fori_loop(0, n_src_per_device, loop_body, 0.0)
            return total_chi2

        # The pmap call remains identical. It shards the data correctly, and the fori_loop
        # will operate on the per-device shards.
        pmapped_calculator = jax.pmap(
            _pmapped_chi2_calculator,
            in_axes=(
                None,  # beam_params_phys - shared
                {"yoff": 0, "xoff": 0, "flux_correction": 0},  # source_params_phys - sharded
                0,     # initial_amplitudes - sharded
                0,     # data_fft - sharded
                None,  # noise_psd - shared
            ),
            axis_name="devices",
        )
        return pmapped_calculator

    def objective_function(self, params_logit, extra_args=None):
        """
        Total loss function

        Parameters:
        -----------
        params_logit : dict
            Pytree of parameters in logit space with structure:
            {"beams": {band: {...}}, "sources": {"yoff": ..., "xoff": ..., "flux_correction": ...}}
        extra_args : optional
            Unused parameter needed for optimistix compatibility
        """
        # Convert logit parameters to physical space
        params_phys = self.param_manager.to_physical(params_logit)

        # Execute the pre-compiled pmapped function. JAX handles sharding the data.
        per_device_chi2_sums = self._pmapped_chi2_calculator(
            params_phys["beams"],
            params_phys["sources"],
            self.gaussfit_initial_amp_jax_padded,
            self.maps_fft_jax_padded,
            self.noise_psd_jax,
        )

        # Use psum to perform a collective sum of the results from all devices.
        total_chi2 = jax.lax.psum(per_device_chi2_sums, axis_name="devices")

        # In a multi-host environment, psum returns the global sum on each host.
        # For single-host, multi-device, this gives the total sum.
        return total_chi2[0] if isinstance(per_device_chi2_sums, jax.Array) else total_chi2

    def _get_individual_chi2s(self, params_logit):
        """Get individual chi2 values for each source (not JIT compiled)."""
        # Convert to physical space
        params_phys = self.param_manager.to_physical(params_logit)

        in_axes = (
            None,  # beam_params_phys - not sharded (same for all sources)
            {"yoff": 0, "xoff": 0, "flux_correction": 0},  # source_params_phys - sharded along axis 0
            0,  # initial_amplitudes - sharded along axis 0
            0,  # data_fft_src - sharded along axis 0
            None,  # noise_psd_src - let's assume it's shared (we'll need to fix this for noise PSDs that are more individual)
        )

        def vmap_friendly_loss(beam_params_phys, source_params_phys, initial_amplitudes, data_fft_src, noise_psd_src):
            # data_fft_src has shape (y, x, band, stokes)
            # noise_psd_src has shape (y, x, band, stokes)
            # initial_amplitudes has shape (band, stokes)
            return self._objective_function_single_source_core(beam_params_phys, source_params_phys, initial_amplitudes, data_fft_src, noise_psd_src)

        # Use the core objective function
        vmap_loss = jax.vmap(vmap_friendly_loss, in_axes=in_axes)

        all_chi2s = vmap_loss(
            params_phys["beams"],
            params_phys["sources"],
            self.gaussfit_initial_amp_jax_padded,
            self.maps_fft_jax_padded,
            self.noise_psd_jax,
        )
        return all_chi2s

    def run_fit(self):
        """Run ML optimization using optimistix for both single and multi-band configurations."""
        print(f"Starting optimization with max_steps={self.config.n_steps}...")

        solver = optx.BFGS(rtol=1e-12, atol=1e-12)
        y0 = self.params
        sol = optx.minimise(self.objective_function, solver, y0, max_steps=self.config.n_steps, throw=False)

        if sol.result != optx.RESULTS.successful:
            print(f"ERROR! BFGS did not converge successfully in {self.config.n_steps} steps.")
            print(f"Final physical params: {self.get_physical_params(sol.value)}")
            raise RuntimeError(optx.RESULTS[sol.result])

        self.params = sol.value

        print(f"Optimization finished after {sol.stats['num_steps']} steps.")
        print(optx.RESULTS[sol.result])

        # Calculate final chi2
        final_chi2s = self._get_individual_chi2s(self.params)
        self.latest_chi2s = np.array(final_chi2s)
        print(f"Final chi2: {np.sum(self.latest_chi2s)}")

        return self.get_physical_params(self.params)

    def get_physical_params(self, params_logit=None):
        """Convert parameters from logit space to physical space."""
        if params_logit is None:
            params_logit = self.params
        return self.param_manager.to_physical(params_logit)

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
        """Unflattens the output of mcmc.get_samples() back into a pytree."""
        # Use the structure of the initial parameters as a template
        template_tree = self.get_physical_params()

        # Get the paths and leaves of the template tree
        leaves, treedef = jax.tree_util.tree_flatten_with_path(template_tree)

        # Reconstruct the list of leaves from the flat samples dict
        new_leaves = []
        for path, _ in leaves:
            # Recreate the flattened key used by numpyro (e.g., 'beams__90GHz__beta_T')
            key = "__".join(str(p.key) for p in path)
            if key in flat_samples:
                new_leaves.append(flat_samples[key])
            else:
                # This should not happen if all params are sampled
                raise KeyError(f"Sample key '{key}' not found in MCMC output.")

        # Use the treedef to reconstruct the pytree
        return jax.tree_util.tree_unflatten(treedef, new_leaves)

    def _nuts_model(self, *args, **kwargs):
        """Numpyro model with uniform priors in physical space."""
        # Sample all parameters in physical space based on the manager's spec tree
        params_phys = self._sample_params_phys_uniform()

        # Convert to logit space for the objective function
        params_logit = self.param_manager.to_logit(params_phys)

        # Calculate chi2 and add to likelihood
        chi2_total = self.objective_function(params_logit, None)
        norm = jnp.asarray(self.config.chi2_normalization, dtype=jnp.float32)
        numpyro.factor("likelihood", -0.5 * norm * chi2_total)

    def _sample_params_phys_uniform(self):
        """Sample all physical parameters using uniform priors from the ParameterManager spec."""
        # Get the pytree structure with initial physical values (for shapes)
        initial_params_phys = self.param_manager.get_initial_params(in_logit_space=False)
        spec_tree = self.param_manager._spec_tree

        def sample_leaf(path, leaf_value):
            # Retrieve bounds for the current parameter from the spec tree
            bounds = self.param_manager._get_bounds_from_path(path)
            low, high = jnp.asarray(bounds[0], dtype=jnp.float32), jnp.asarray(bounds[1], dtype=jnp.float32)

            # Create a distribution with the same shape as the parameter
            uniform_dist = dist.Uniform(low=jnp.full_like(leaf_value, low), high=jnp.full_like(leaf_value, high))

            # Create a unique name for numpyro sampling (e.g., "beams__90GHz__beta_T")
            name = "__".join(str(p.key) for p in path)

            return numpyro.sample(name, uniform_dist.to_event(leaf_value.ndim))

        # Use tree_map_with_path to apply the sampling function to each leaf
        return jax.tree_util.tree_map_with_path(sample_leaf, initial_params_phys)

    def _nuts_init(self):
        """Initialize NUTS at current physical parameter values."""
        # Get the current best-fit parameters in physical space
        phys = self.get_physical_params(self.params)

        # Flatten the pytree into a dictionary with numpyro-compatible keys
        flat_phys_params, _ = jax.tree_util.tree_flatten_with_path(phys)
        init_vals = {"__".join(str(p.key) for p in path): jnp.asarray(val, dtype=jnp.float32) for path, val in flat_phys_params}

        return init_to_value(values=init_vals)

    def create_beam_profile_maps(self, best_fit_params):
        """Create centered beam maps for radial profile plotting."""
        beam_T_maps = {}
        beam_P_maps = {}
        for band in self.bands:
            beam_T_map, beam_P_map = self.beam_models[band].evaluate_beam_maps(best_fit_params["beams"][band], 0.0, 0.0)
            beam_T_maps[band] = beam_T_map
            beam_P_maps[band] = beam_P_map
        return beam_T_maps, beam_P_maps
