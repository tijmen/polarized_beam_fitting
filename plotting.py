"""
Plotting and visualization functions for polarized beam analysis.

Contains functions for creating diagnostic plots, beam profiles, and analysis visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from .utils import compute_2d_asd, safe_filename
from spt3g import core


class BeamPlotter:
    """
    Class for creating various plots and visualizations for beam analysis.
    """

    def __init__(self, fitter, output_dir=None):
        """
        Initialize the plotter.

        Parameters:
        -----------
        fitter : PolarizedBeamFitter, or BootstrapBeamFitter
            The fitted beam fitter object
        output_dir : str, optional
            Output directory for plots. If None, uses fitter's config.
        """
        self.fitter = fitter

        # Handle both regular fitter and bootstrap fitter
        if hasattr(fitter, "base_fitter"):
            # Bootstrap fitter - delegate to the base fitter for data access
            self.base_fitter = fitter.base_fitter
            self.output_dir = output_dir or fitter.config.output_dir
        else:
            # Regular fitter
            self.base_fitter = fitter
            self.output_dir = output_dir or fitter.config.output_dir

        # Detect if this is a multi-band fitter
        self.is_multiband = hasattr(self.base_fitter, "beam_models") and isinstance(self.base_fitter.beam_models, dict)

        # Get band information
        if self.is_multiband:
            self.bands = list(self.base_fitter.beam_models.keys())
            self.primary_band = self.bands[1] if len(self.bands) > 1 else self.bands[0]  # Use 150GHz as primary
        else:
            self.bands = [getattr(self.base_fitter, "band", "unknown")]
            self.primary_band = self.bands[0]

        os.makedirs(self.output_dir, exist_ok=True)

    def _get_beam_model(self, band=None):
        """Get the appropriate beam model for the given band."""
        if self.is_multiband:
            if band is None:
                band = self.primary_band
            return self.base_fitter.beam_models[band]
        else:
            return self.base_fitter.beam_model

    def _get_band_suffix(self, band=None):
        """Get the band suffix for filenames."""
        if self.is_multiband:
            if band is None:
                band = self.primary_band
            return band.replace("GHz", "")
        else:
            return getattr(self.base_fitter, "band", "unknown").replace("GHz", "")

    def _get_fit_params_for_band(self, best_fit_params, band=None):
        """Extract fit parameters for a specific band from multi-band results."""
        if self.is_multiband:
            if band is None:
                band = self.primary_band
            # For multi-band, extract band-specific parameters
            band_params = best_fit_params["bands"][band]
            # Combine shared and band-specific parameters
            fit_params = {
                "beam": band_params["beam"],
                "sources": {
                    "y_offset": best_fit_params["shared"]["y_offset"],
                    "x_offset": best_fit_params["shared"]["x_offset"],
                    "t_amp_factor": band_params["t_amp_factor"],
                    "q_amp_factor": band_params["q_amp_factor"],
                    "u_amp_factor": band_params["u_amp_factor"],
                },
            }
            return fit_params
        else:
            return best_fit_params

    def plot_template_projection_analysis(self, best_fit_params, skip_sources=None, save=True):
        """
        Plot template projection analysis using the brightest source as template.

        This method uses the brightest source as a template to project out of
        extended/problematic sources to analyze their residual structure.

        Parameters:
        -----------
        best_fit_params : dict
            Best-fit parameters from optimization
        skip_sources : list, optional
            List of source names to analyze. If None, uses config.skip_sources
        save : bool
            Whether to save the plot

        Returns:
        --------
        str or None
            Filename if saved, None otherwise
        """
        # For multi-band, use primary band for template analysis
        if self.is_multiband:
            band = self.primary_band
            band_suffix = self._get_band_suffix(band)
            print(f"\n--- Template Projection Analysis for {band_suffix} (multi-band fitter) ---")
        else:
            band_suffix = self._get_band_suffix()
            print(f"\n--- Template Projection Analysis for {band_suffix} ---")

        if skip_sources is None:
            skip_sources = self.fitter.config.skip_sources

        if not skip_sources:
            print("No skip_sources specified for template projection analysis")
            return None

        # Get data maps and source information
        data_maps = self.fitter.maps_numpy
        source_ids = self.fitter.source_ids

        # Multi-band: calculate T amplitudes from the primary band
        if self.is_multiband:
            t_amps = []
            # Get band index for primary band
            band_idx = self.fitter.bands.index(band)
            for i, source_id in enumerate(source_ids):
                # Get T amplitude for this source in the primary band using array format
                t_amp_initial = self.fitter.initial_amplitudes_array[i, band_idx, 0]  # T is index 0
                t_amp_factor = best_fit_params["bands"][band]["t_amp_factor"][i]
                t_amps.append(t_amp_initial * t_amp_factor)
            t_amps = np.array(t_amps)
        else:
            # Single band fitter - use array format with band index 0
            t_amps = self.fitter.initial_amplitudes_array[:, 0, 0] * best_fit_params["sources"]["t_amp_factor"]

        # Find brightest source as template
        brightest_idx = np.argmax(t_amps)
        brightest_source_id = source_ids[brightest_idx]

        # For multi-band, get template from primary band data
        if self.is_multiband:
            template_map = data_maps[brightest_source_id][band]["T"]
        else:
            template_map = data_maps[brightest_source_id]["T"]

        print(f"Using brightest source as template: {brightest_source_id}")
        print(f"Template amplitude: {t_amps[brightest_idx]:.1f} μK")

        # Find skip_sources in the data
        skip_sources_data = []
        for source in skip_sources:
            if self.is_multiband:
                # For multi-band, construct source ID with primary band
                source_id = f"CoaddSPT-S {source}-{band}"
            else:
                source_id = "CoaddSPT-S " + source + "-" + self.fitter.band

            if source_id in source_ids:
                idx = source_ids.index(source_id)
                skip_sources_data.append((source_id, idx, source))
                print(f"Found skip source: {source_id}")

        if not skip_sources_data:
            print("No skip sources found in the data")
            return None

        print(f"\nFound {len(skip_sources_data)} skip sources in the data")

        # Create plot: rows are skip sources, columns are data and residual after template projection
        n_sources = len(skip_sources_data)
        fig, axes = plt.subplots(n_sources, 2, figsize=(12, 4 * n_sources))
        if n_sources == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f"Template Projection Analysis ({band_suffix})\nTemplate: {brightest_source_id}", fontsize=14)

        for i, (source_id, source_idx, short_name) in enumerate(skip_sources_data):
            # Get data map for this skip source
            if self.is_multiband:
                data_map = data_maps[source_id][band]["T"]
            else:
                data_map = data_maps[source_id]["T"]

            # Project out template: residual = data - α * template
            # Find optimal scaling factor α using least squares
            template_flat = template_map.flatten()
            data_flat = data_map.flatten()

            # α = (template · data) / (template · template)
            alpha = np.dot(template_flat, data_flat) / np.dot(template_flat, template_flat)

            # Create residual map
            residual_map = data_map - alpha * template_map

            # Plot data (left column)
            ax = axes[i, 0]
            data_max = np.max(np.abs(data_map))
            im1 = ax.imshow(data_map, cmap="RdBu_r", origin="lower", vmin=-data_max, vmax=data_max)
            ax.set_title(f"{short_name}\nData T Map", fontsize=10)
            ax.set_ylabel("Y pixel")
            if i == n_sources - 1:  # bottom row
                ax.set_xlabel("X pixel")
            plt.colorbar(im1, ax=ax, label="T (mK)", fraction=0.046, pad=0.04)

            # Plot residual (right column)
            ax = axes[i, 1]
            residual_max = 0.5  # Fixed scale for comparison
            im2 = ax.imshow(residual_map, cmap="RdBu_r", origin="lower", vmin=-residual_max, vmax=residual_max)
            ax.set_title(f"{short_name}\nResidual after Template Projection\n(α = {alpha:.3f})", fontsize=10)
            if i == n_sources - 1:  # bottom row
                ax.set_xlabel("X pixel")
            plt.colorbar(im2, ax=ax, label="Residual (mK)", fraction=0.046, pad=0.04)

            # Print statistics
            data_peak = np.max(np.abs(data_map))
            residual_rms = np.std(residual_map)
            residual_peak = np.max(np.abs(residual_map))
            reduction_factor = data_peak / residual_peak if residual_peak > 0 else np.inf

            print(f"\n{short_name}:")
            print(f"  Data peak: {data_peak:.1f} μK")
            print(f"  Template scaling: α = {alpha:.3f}")
            print(f"  Residual RMS: {residual_rms:.1f} μK")
            print(f"  Residual peak: {residual_peak:.1f} μK")
            print(f"  Peak reduction factor: {reduction_factor:.1f}x")

        plt.tight_layout()

        if save:
            plot_filename = os.path.join(self.output_dir, f"template_projection_analysis_{band_suffix}.png")
            plt.savefig(plot_filename, dpi=300)
            plt.close(fig)
            print(f"\nSaved template projection analysis plot to: {plot_filename}")
            print(f"=== Template projection analysis complete for {band_suffix} ===")
            return plot_filename
        else:
            plt.show()
            return None

    def plot_beam_profiles(self, best_fit_params, save=True, band=None):
        """
        Plot radial beam profiles and T-P beam difference with optional bootstrap uncertainties.

        Parameters:
        -----------
        best_fit_params : dict
            Best-fit parameters from optimization (or bootstrap results if available)
        save : bool
            Whether to save the plot
        band : str, optional
            For multi-band fitters, specify which band to plot. If None, plots primary band.

        Returns:
        --------
        str or None
            Filename if saved, None otherwise
        """
        print("\n--- Generating Beam Profiles and T-P Beam Difference ---")

        # Check if we have bootstrap results
        has_bootstrap = hasattr(self.fitter, "bootstrap_results") and self.fitter.bootstrap_results is not None
        if has_bootstrap:
            print("Bootstrap results detected - including uncertainty bands")
            fit_params = best_fit_params.get("original_fit", best_fit_params)
        else:
            fit_params = best_fit_params

        if self.is_multiband:
            # For multi-band, create plots for each band
            if band is None:
                # Plot all bands
                return self._plot_multiband_beam_profiles(fit_params, save)
            else:
                # Plot specific band
                return self._plot_single_band_beam_profiles(fit_params, save, band)
        else:
            # Single band fitter
            return self._plot_single_band_beam_profiles(fit_params, save)

    def _plot_single_band_beam_profiles(self, best_fit_params, save=True, band=None):
        """Plot beam profiles for a single band."""
        # Get fit parameters for this band
        fit_params = self._get_fit_params_for_band(best_fit_params, band)
        beam_model = self._get_beam_model(band)
        band_suffix = self._get_band_suffix(band)

        # Use the unified API
        r_fine, profile_T_fine, profile_P_fine, info = beam_model.get_profiles_for_plotting(fit_params)

        print(f"Beam profiles for {band_suffix} (peak normalized):")
        print(f"  T-beam: peak = {np.max(profile_T_fine):.4f}")
        print(f"  P-beam: peak = {np.max(profile_P_fine):.4f}")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        model_type = self.fitter.config.beam_model_type.replace("_", "-").title()
        fig.suptitle(f"Best-Fit {model_type} Beam Model ({band_suffix})", fontsize=16)

        # Top panel: Final Beam profiles
        t_label = info.get("t_label", "Best-Fit T-Beam Profile")
        p_label = info.get("p_label", "Best-Fit P-Beam Profile")
        ylabel = info.get("ylabel", "Beam Amplitude")

        ax1.plot(r_fine, profile_T_fine, label=t_label, lw=3, color="C0", zorder=10)
        ax1.plot(r_fine, profile_P_fine, label=p_label, lw=3, linestyle="--", color="C1", zorder=10)
        ax1.axhline(0, color="black", lw=0.5, zorder=1)

        y_min = min(np.min(profile_T_fine), np.min(profile_P_fine))
        y_max = max(np.max(profile_T_fine), np.max(profile_P_fine))
        y_range = y_max - y_min
        ax1.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        ax1.grid(True, which="both", linestyle=":", alpha=0.5)
        ax1.set_ylabel(ylabel, fontsize=12)
        ax1.set_title(f"Reconstructed Beam Profiles", fontsize=14)
        ax1.legend(fontsize=11)

        # Bottom panel: T beam minus P beam
        beam_difference = profile_T_fine - profile_P_fine
        ax2.plot(r_fine, beam_difference, lw=3, color="C2", label="T-Beam - P-Beam")
        ax2.axhline(0, color="black", lw=0.5)
        ax2.set_xlabel("Radius [arcmin]", fontsize=12)
        ax2.set_ylabel("Amplitude Difference", fontsize=12)
        ax2.set_title("T-Beam minus P-Beam", fontsize=14)
        ax2.set_xlim(-0.2, 10.0)
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(fontsize=11)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save:
            plot_filename = os.path.join(self.output_dir, f"beam_profile_{band_suffix}.png")
            plt.savefig(plot_filename, dpi=200)
            plt.close(fig)
            print(f"Saved beam profile plot to: {plot_filename}")
            return plot_filename
        else:
            plt.show()
            return None

    def _plot_multiband_beam_profiles(self, best_fit_params, save=True):
        """Plot beam profiles for all bands in a single figure."""
        fig, axes = plt.subplots(len(self.bands), 2, figsize=(14, 4 * len(self.bands)), sharex=True)
        if len(self.bands) == 1:
            axes = axes.reshape(1, -1)

        model_type = self.fitter.config.beam_model_type.replace("_", "-").title()
        fig.suptitle(f"Multi-Band {model_type} Beam Models", fontsize=16)

        colors = ["C0", "C1", "C2"]

        for i, band in enumerate(self.bands):
            # Get fit parameters for this band
            fit_params = self._get_fit_params_for_band(best_fit_params, band)
            beam_model = self._get_beam_model(band)
            band_suffix = self._get_band_suffix(band)

            # Get beam profiles
            r_fine, profile_T_fine, profile_P_fine, info = beam_model.get_profiles_for_plotting(fit_params)

            # Top panel: Beam profiles
            ax = axes[i, 0]
            ax.plot(r_fine, profile_T_fine, label=f"T-Beam ({band_suffix})", lw=2, color=colors[i])
            ax.plot(r_fine, profile_P_fine, label=f"P-Beam ({band_suffix})", lw=2, linestyle="--", color=colors[i], alpha=0.7)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_ylabel("Beam Amplitude", fontsize=12)
            ax.set_title(f"{band_suffix} Beam Profiles", fontsize=14)
            ax.grid(True, which="both", linestyle=":", alpha=0.5)
            ax.legend(fontsize=10)

            # Bottom panel: T-P difference
            ax = axes[i, 1]
            beam_difference = profile_T_fine - profile_P_fine
            ax.plot(r_fine, beam_difference, lw=2, color=colors[i], label=f"T-P ({band_suffix})")
            ax.axhline(0, color="black", lw=0.5)
            ax.set_ylabel("Amplitude Difference", fontsize=12)
            ax.set_title(f"{band_suffix} T-Beam minus P-Beam", fontsize=14)
            ax.set_xlim(-0.2, 10.0)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(fontsize=10)

            if i == len(self.bands) - 1:  # bottom row
                ax.set_xlabel("Radius [arcmin]", fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save:
            plot_filename = os.path.join(self.output_dir, "beam_profile_multiband.png")
            plt.savefig(plot_filename, dpi=200)
            plt.close(fig)
            print(f"Saved multi-band beam profile plot to: {plot_filename}")
            return plot_filename
        else:
            plt.show()
            return None

    def plot_basis_diagnostics(self, save=True, band=None):
        """
        Plot diagnostic information about the orthonormal basis functions.
        Only works for B-spline beam models.

        Parameters:
        -----------
        save : bool
            Whether to save the plot
        band : str, optional
            For multi-band fitters, specify which band to plot. If None, plots primary band.

        Returns:
        --------
        str or None
            Filename if saved, None otherwise
        """
        if self.fitter.config.beam_model_type != "b_spline":
            print(f"Skipping basis diagnostics for {self.fitter.config.beam_model_type} beam model")
            return None

        if self.is_multiband:
            if band is None:
                # For multi-band, plot primary band
                band = self.primary_band
            beam_model = self._get_beam_model(band)
            band_suffix = self._get_band_suffix(band)
        else:
            beam_model = self._get_beam_model()
            band_suffix = self._get_band_suffix()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot particular solution
        ax = axes[0, 0]
        r = np.array(beam_model.r_fine_jax)
        particular = np.array(beam_model.particular_func_jax)
        ax.plot(r, particular, "k-", linewidth=2)
        ax.set_title("Particular Solution (satisfies boundary conditions)")
        ax.set_xlabel("Radius (arcmin)")
        ax.set_ylabel("B(r)")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

        # Plot first few orthonormal basis functions
        ax = axes[0, 1]
        ortho_funcs = np.array(beam_model.ortho_basis_funcs_jax)
        n_plot = min(5, ortho_funcs.shape[1])
        for i in range(n_plot):
            ax.plot(r, ortho_funcs[:, i], label=f"φ_{i}")
        ax.set_title("First Few Orthonormal Basis Functions")
        ax.set_xlabel("Radius (arcmin)")
        ax.set_ylabel("φ(r)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        # Plot example beam profiles
        ax = axes[1, 0]
        ax.plot(r, particular, "k-", linewidth=2, label="Particular only")

        # Add some basis functions
        example_coeffs = [([1, 0, 0, 0, 0], "Mode 0 only"), ([0, 1, 0, 0, 0], "Mode 1 only"), ([0.5, 0.3, 0.1, 0, 0], "Mixed modes")]

        for coeffs, label in example_coeffs:
            coeffs_array = np.zeros(ortho_funcs.shape[1])
            coeffs_array[: len(coeffs)] = coeffs
            profile = particular + ortho_funcs @ coeffs_array
            ax.plot(r, profile, "--", label=label, alpha=0.7)

        ax.set_title("Example Beam Profiles")
        ax.set_xlabel("Radius (arcmin)")
        ax.set_ylabel("B(r)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(-0.1, 1.2)

        # Verify orthonormality visually
        ax = axes[1, 1]
        n_check = min(10, ortho_funcs.shape[1])
        gram_visual = np.zeros((n_check, n_check))

        # Compute inner products with proper weight
        weight = r.copy()
        weight[0] = weight[1] * 0.5

        for i in range(n_check):
            for j in range(n_check):
                gram_visual[i, j] = np.trapz(ortho_funcs[:, i] * ortho_funcs[:, j] * weight, r)

        im = ax.imshow(gram_visual, cmap="RdBu_r", vmin=-0.1, vmax=1.1)
        ax.set_title("Gram Matrix of Basis Functions")
        ax.set_xlabel("Basis function index")
        ax.set_ylabel("Basis function index")
        plt.colorbar(im, ax=ax)

        plt.tight_layout()

        if save:
            plot_filename = os.path.join(self.output_dir, f"orthonormal_basis_{band_suffix}.png")
            plt.savefig(plot_filename, dpi=150)
            plt.close(fig)
            print(f"Saved basis diagnostics plot to: {plot_filename}")
            return plot_filename
        else:
            plt.show()
            return None

    def plot_source_diagnostics(self, best_fit_params, n_sources=None, central_crop=None, save=True):
        """
        Plot data/model/residual maps for the sources with highest chi-squared values.

        Parameters:
        -----------
        best_fit_params : dict
            Best-fit parameters from optimization
        n_sources : int, str, or None
            Number of highest chi-squared sources to plot. Can be:
            - An integer: plot that many sources
            - "all": plot all sources
            - 0: plot no sources (skip diagnostic plots)
            - None: use the value from config.n_diagnostic_plots
        central_crop : int
            Option to crop to central `central_crop` x `central_crop` pixels (default: None, no cropping)
        save : bool
            Whether to save the plots

        Returns:
        --------
        list
            List of filenames if saved, empty list otherwise
        """
        # Use config value if not specified
        if n_sources is None:
            n_sources = self.fitter.config.n_diagnostic_plots

        # Handle special cases
        if n_sources == 0:
            print("Skipping source diagnostic plots (n_diagnostic_plots = 0)")
            return []

        # Determine actual number of sources to plot
        total_sources = len(self.fitter.source_ids)
        if isinstance(n_sources, str) and n_sources.lower() == "all":
            n_to_plot = total_sources
            print(f"\n--- Generating Data/Model/Residual Maps for All {n_to_plot} Sources ---")
        else:
            n_to_plot = min(int(n_sources), total_sources)
            print(f"\n--- Generating Data/Model/Residual Maps for Top {n_to_plot} Highest Chi2 Sources ---")

        if central_crop is not None:
            print(f"Using central {central_crop}x{central_crop} pixel crop")

        data_maps = self.fitter.maps_numpy
        model_maps = self.fitter.create_final_model_maps(best_fit_params)

        # Sort sources by polarization amplitude
        p_amp_sources = []
        for i, source_id in enumerate(self.fitter.source_ids):
            if self.is_multiband:
                # For multi-band, use primary band for ranking
                band = self.primary_band
                band_idx = self.fitter.bands.index(band)
                t_amp_initial = self.fitter.initial_amplitudes_array[i, band_idx, 0]  # T is index 0
                q_amp_initial = self.fitter.initial_amplitudes_array[i, band_idx, 1]  # Q is index 1
                u_amp_initial = self.fitter.initial_amplitudes_array[i, band_idx, 2]  # U is index 2
                t_amp = t_amp_initial * best_fit_params["bands"][band]["t_amp_factor"][i]
                q_amp = q_amp_initial * best_fit_params["bands"][band]["q_amp_factor"][i]
                u_amp = u_amp_initial * best_fit_params["bands"][band]["u_amp_factor"][i]
                p_amp = float(np.sqrt(q_amp**2 + u_amp**2))
            else:
                # Single band fitter - use array format with band index 0
                t_amp = float(self.fitter.initial_amplitudes_array[i, 0, 0])
                q_amp = float(self.fitter.initial_amplitudes_array[i, 0, 1] * best_fit_params["sources"]["q_amp_factor"][i])
                u_amp = float(self.fitter.initial_amplitudes_array[i, 0, 2] * best_fit_params["sources"]["u_amp_factor"][i])
                p_amp = float(np.sqrt(q_amp**2 + u_amp**2))
            p_amp_sources.append((p_amp, source_id, i))

        p_amp_sources.sort(key=lambda x: x[0], reverse=True)

        print("Top 10 sources by polarization amplitude:")
        for rank, (p_amp, source_id, idx) in enumerate(p_amp_sources[:10]):
            print(f"  {rank + 1:2d}. {source_id}: p_amp = {p_amp:.1f} mK")

        filenames = []
        for rank in range(1, n_to_plot + 1):
            if rank <= len(p_amp_sources):
                p_amp, source_id, idx = p_amp_sources[rank - 1]
                print(f"\nCreating model/data/residual plot for rank #{rank} source: {source_id} (p_amp = {p_amp:.1f} mK)")
                filename = self._create_source_diagnostic_plot(source_id, rank, data_maps, model_maps, central_crop, save)
                if filename:
                    filenames.append(filename)
                    print(f"Saved model/data/residual plot to: {filename}")

        return filenames

    def _create_source_diagnostic_plot(self, source_id, rank, data_maps, model_maps, central_crop=None, save=True):
        """Create a diagnostic plot for a single source."""
        if self.is_multiband:
            # For multi-band, use primary band for plotting
            band = self.primary_band
            data = data_maps[source_id][band]
            model = model_maps[source_id][band]
            band_suffix = self._get_band_suffix(band)
        else:
            data = data_maps[source_id]
            model = model_maps[source_id]
            band_suffix = self._get_band_suffix()

        residual = {k: data[k] - model[k] for k in data}

        # Apply central crop if requested
        if central_crop is not None:
            crop_size = central_crop
            for map_dict in [data, model, residual]:
                for stokes in map_dict:
                    map_shape = map_dict[stokes].shape
                    center_y, center_x = map_shape[0] // 2, map_shape[1] // 2
                    half_crop = crop_size // 2
                    y_start = max(0, center_y - half_crop)
                    y_end = min(map_shape[0], center_y + half_crop)
                    x_start = max(0, center_x - half_crop)
                    x_end = min(map_shape[1], center_x + half_crop)
                    map_dict[stokes] = map_dict[stokes][y_start:y_end, x_start:x_end]

        fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
        crop_suffix = f" (Central {central_crop}x{central_crop})" if central_crop else ""
        fig.suptitle(f"Data/Model/Residual Maps for Source #{rank}: {source_id}{crop_suffix} ({band_suffix})", fontsize=16)

        maps_to_plot = {"Data": data, "Model": model, "Residual": residual}

        row_labels = ["T", "Q", "U"]
        col_labels = ["Data", "Model", "Residual"]

        for i, map_type in enumerate(row_labels):
            # Determine color limits for Data and Model from the Data map
            vmax = np.max(np.abs(data[map_type]))
            vmin = -vmax

            # Determine color limits for Residual
            res_vmax = np.max(np.abs(residual[map_type]))

            for j, plot_type in enumerate(col_labels):
                ax = axes[i, j]

                if plot_type == "Residual":
                    im = ax.imshow(maps_to_plot[plot_type][map_type], cmap="RdBu_r", vmin=-res_vmax, vmax=res_vmax)
                else:
                    im = ax.imshow(maps_to_plot[plot_type][map_type], cmap="viridis", vmin=vmin, vmax=vmax)

                fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04, label="mK")

                if i == 0:
                    ax.set_title(plot_type, fontsize=14)
                if j == 0:
                    ax.set_ylabel(map_type, fontsize=14, rotation=0, labelpad=20)

                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save:
            safe_source_id = safe_filename(source_id)
            plot_filename = os.path.join(self.output_dir, f"diagnostic_maps_{band_suffix}_{safe_source_id}.png")
            plt.savefig(plot_filename, dpi=150)
            plt.close(fig)
            return plot_filename
        else:
            plt.show()
            return None

    def plot_asd_analysis(self, best_fit_params, save=True):
        """
        Plot 2D amplitude spectral density analysis for the top source.

        Parameters:
        -----------
        best_fit_params : dict
            Best-fit parameters from optimization
        save : bool
            Whether to save the plot

        Returns:
        --------
        str or None
            Filename if saved, None otherwise
        """
        print("\n--- Generating ASD Analysis for Top Source ---")

        # Get data and model maps
        data_maps = self.fitter.maps_numpy
        model_maps = self.fitter.create_final_model_maps(best_fit_params)

        # Get the top source by T amplitude
        t_amps = []
        for i, source_id in enumerate(self.fitter.source_ids):
            if self.is_multiband:
                # For multi-band, use primary band for ranking
                band = self.primary_band
                band_idx = self.fitter.bands.index(band)
                t_amp_initial = self.fitter.initial_amplitudes_array[i, band_idx, 0]  # T is index 0
                t_amp_factor = best_fit_params["bands"][band]["t_amp_factor"][i]
                t_amp = float(t_amp_initial * t_amp_factor)
            else:
                # Single band fitter - use array format with band index 0
                t_amp = float(self.fitter.initial_amplitudes_array[i, 0, 0])
            t_amps.append((t_amp, source_id, i))
        t_amps.sort(key=lambda x: x[0], reverse=True)

        t_amp, top_source_id, idx = t_amps[0]
        print(f"Analyzing top source: {top_source_id} (t_amp = {t_amp:.1f} mK)")

        # Get data, model, and residual for top source
        if self.is_multiband:
            # For multi-band, use primary band for analysis
            band = self.primary_band
            data_top = data_maps[top_source_id][band]
            model_top = model_maps[top_source_id][band]
            band_suffix = self._get_band_suffix(band)
        else:
            data_top = data_maps[top_source_id]
            model_top = model_maps[top_source_id]
            band_suffix = self._get_band_suffix()

        residual_top = {k: data_top[k] - model_top[k] for k in data_top}

        # Create ASD analysis plot: 3 rows (T, Q, U) x 4 columns (Data, Model, Residual, Residual/Noise)
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        fig.suptitle(f"2D Amplitude Spectral Density for Top Source: {top_source_id} ({band_suffix})", fontsize=16)

        stokes_params = ["T", "Q", "U"]

        for i, stokes in enumerate(stokes_params):
            # Compute ASDs for data, model, and residual
            asd_data = compute_2d_asd(data_top[stokes])
            asd_model = compute_2d_asd(model_top[stokes])
            asd_residual = compute_2d_asd(residual_top[stokes])

            # Get noise PSD for this Stokes parameter and source
            noise_psd_key = {"T": "TT", "Q": "QQ", "U": "UU"}[stokes]

            if self.fitter.noise_psd_calculator.is_individual_psds():
                # Individual noise PSD for this specific source
                noise_psd = self.fitter.noise_psd_py[idx][noise_psd_key]
            else:
                # Global noise PSD for all sources
                noise_psd = self.fitter.noise_psd_py[noise_psd_key]

            # Compute residual/noise ratio (convert PSD to ASD by taking sqrt)
            noise_asd = np.fft.fftshift(np.sqrt(noise_psd))
            asd_residual_over_noise = asd_residual / noise_asd

            # Convert to log scale for better visualization
            asd_data_log = np.log10(asd_data + 1e-20)
            asd_model_log = np.log10(asd_model + 1e-20)
            asd_residual_log = np.log10(asd_residual + 1e-20)
            asd_residual_over_noise_log = np.log10(asd_residual_over_noise + 1e-20)

            # Determine common color scale for data and model
            vmin_common = min(np.min(asd_data_log), np.min(asd_model_log))
            vmax_common = max(np.max(asd_data_log), np.max(asd_model_log))

            # Determine color scale for residual
            vmin_resid = np.min(asd_residual_log)
            vmax_resid = np.max(asd_residual_log)

            # Determine color scale for residual/noise
            vmin_resid_noise = np.min(asd_residual_over_noise_log)
            vmax_resid_noise = np.max(asd_residual_over_noise_log)

            # Plot data
            ax = axes[i, 0]
            im = ax.imshow(asd_data_log, cmap="viridis", origin="lower", vmin=vmin_common, vmax=vmax_common)
            if i == 0:
                ax.set_title("Data", fontsize=14)
            ax.set_ylabel(f"{stokes}", fontsize=14, rotation=0, labelpad=20)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Plot model
            ax = axes[i, 1]
            im = ax.imshow(asd_model_log, cmap="viridis", origin="lower", vmin=vmin_common, vmax=vmax_common)
            if i == 0:
                ax.set_title("Model", fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Plot residual
            ax = axes[i, 2]
            im = ax.imshow(asd_residual_log, cmap="viridis", origin="lower", vmin=vmin_resid, vmax=vmax_resid)
            if i == 0:
                ax.set_title("Residual", fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Plot residual/noise
            ax = axes[i, 3]
            im = ax.imshow(asd_residual_over_noise_log, cmap="RdBu_r", origin="lower", vmin=vmin_resid_noise, vmax=vmax_resid_noise)
            if i == 0:
                ax.set_title("Residual/Noise", fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save:
            safe_source_id = safe_filename(top_source_id)
            asd_plot_filename = os.path.join(self.output_dir, f"asd_2d_{band_suffix}_{safe_source_id}.png")
            plt.savefig(asd_plot_filename, dpi=150)
            plt.close(fig)
            print(f"Saved 2D ASD analysis plot to: {asd_plot_filename}")

            # Print some statistics
            self._print_asd_statistics(top_source_id, data_top, model_top, residual_top, idx)

            return asd_plot_filename
        else:
            plt.show()
            return None

    def _print_asd_statistics(self, source_id, data_top, model_top, residual_top, idx):
        """Print ASD statistics for a source."""
        print(f"\n2D ASD Statistics for {source_id}:")
        stokes_params = ["T", "Q", "U"]

        for stokes in stokes_params:
            asd_data = compute_2d_asd(data_top[stokes])
            asd_model = compute_2d_asd(model_top[stokes])
            asd_residual = compute_2d_asd(residual_top[stokes])

            # Get noise PSD for this Stokes parameter and source
            noise_psd_key = {"T": "TT", "Q": "QQ", "U": "UU"}[stokes]

            if self.fitter.noise_psd_calculator.is_individual_psds():
                # Individual noise PSD for this specific source
                noise_psd = self.fitter.noise_psd_py[idx][noise_psd_key]
            else:
                # Global noise PSD for all sources
                noise_psd = self.fitter.noise_psd_py[noise_psd_key]

            # Compute noise ASD for comparison
            noise_asd = np.fft.fftshift(np.sqrt(noise_psd))

            # Total power (sum of squared amplitudes)
            total_power_data = np.sum(asd_data**2)
            total_power_model = np.sum(asd_model**2)
            total_power_resid = np.sum(asd_residual**2)
            total_power_noise = np.sum(noise_asd**2)

            # Peak amplitude
            peak_data = np.max(asd_data)
            peak_model = np.max(asd_model)
            peak_resid = np.max(asd_residual)
            peak_noise = np.max(noise_asd)

            print(f"  {stokes}: Total power (data/model/resid/noise): {total_power_data:.2e}/{total_power_model:.2e}/{total_power_resid:.2e}/{total_power_noise:.2e}")
            print(f"       Peak amplitude (data/model/resid/noise): {peak_data:.2e}/{peak_model:.2e}/{peak_resid:.2e}/{peak_noise:.2e}")

            # Signal-to-noise ratios
            snr_data = total_power_data / total_power_noise if total_power_noise > 0 else 0
            snr_resid = total_power_resid / total_power_noise if total_power_noise > 0 else 0
            print(f"       SNR (data/residual): {snr_data:.2f}/{snr_resid:.2f}")


def create_diagnostic_plots(fitter, best_fit_params, output_dir=None, include_template_analysis=False, central_crop=None):
    """
    Convenience function to create all diagnostic plots.

    Parameters:
    -----------
    fitter : PolarizedBeamFitter or BootstrapBeamFitter
        The fitted beam fitter object
    best_fit_params : dict
        Best-fit parameters from optimization
    output_dir : str, optional
        Output directory for plots
    include_template_analysis : bool, optional
        Whether to include template projection analysis
    central_crop : int, optional
        Option to crop source diagnostic plots to central `central_crop` x `central_crop` pixels (default: None, no cropping)

    Returns:
    --------
    dict
        Dictionary of plot filenames

    Note:
    -----
    The number of diagnostic plots created is controlled by the
    fitter.config.n_diagnostic_plots parameter, which can be:
    - An integer (default 3): plot that many highest chi2 sources
    - "all": plot all sources
    - 0: skip diagnostic plots entirely

    For multi-band fitters, plots are created using the primary band (150GHz) for analysis.
    """
    plotter = BeamPlotter(fitter, output_dir)

    filenames = {}

    # Beam profiles
    filenames["beam_profiles"] = plotter.plot_beam_profiles(best_fit_params)

    # Basis diagnostics
    filenames["basis_diagnostics"] = plotter.plot_basis_diagnostics()

    # Source diagnostics (uses config.n_diagnostic_plots)
    filenames["source_diagnostics"] = plotter.plot_source_diagnostics(best_fit_params, central_crop=central_crop)

    # ASD analysis
    if not plotter.is_multiband:
        filenames["asd_analysis"] = plotter.plot_asd_analysis(best_fit_params)

    # Template projection analysis (optional)
    if include_template_analysis:
        filenames["template_projection"] = plotter.plot_template_projection_analysis(best_fit_params)

    print("\n--- All plotting complete. ---")

    return filenames
