"""
Plotting and visualization functions for polarized beam analysis.

Contains functions for creating diagnostic plots, beam profiles, and analysis visualizations.
"""

import os

import arviz as az
import corner
import matplotlib.pyplot as plt
import numpy as np

from .utils import safe_filename


def _bin_mean(x, y, edges):
    """Helper function for binning data."""
    # x, y: 1D arrays of equal length
    # returns bin centers and mean(y) per bin, NaN where empty
    idx = np.digitize(x, edges) - 1  # bins 0..nbins-1
    nbins = len(edges) - 1
    valid = (idx >= 0) & (idx < nbins) & np.isfinite(y)
    counts = np.bincount(idx[valid], minlength=nbins)
    sums = np.bincount(idx[valid], weights=y[valid], minlength=nbins)
    means = np.full(nbins, np.nan, dtype=float)
    nz = counts > 0
    means[nz] = sums[nz] / counts[nz]
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, means, counts


def bin_ell_radial(ell_radial, values, ellmax, nbins=50):
    """Bin values radially in ell space."""
    # ell_radial, values can be 1D or broadcastable 2D (ℓ×φ); flatten
    x = np.asarray(ell_radial).ravel()
    y = np.asarray(values).ravel()
    edges = np.linspace(0.0, ellmax, nbins + 1)
    return _bin_mean(x, y, edges)


def _ensure_src_axis(a):
    """Return (arr_with_src, has_src). If no src axis, add a leading singleton."""
    a = np.asarray(a)
    if a.ndim == 6:
        # (ℓ, φ, b, s, b, s) -> add src axis
        return a[None, ...], False
    elif a.ndim == 7:
        return a, True
    else:
        raise ValueError("Unexpected array rank for covariance-like tensor.")


def _field_mean(a):
    """Mean over src if present, otherwise pass-through."""
    a_src, has_src = _ensure_src_axis(a)
    m = np.nanmean(a_src, axis=0)
    return m  # shape: (ℓ, φ, b, s, b, s)


def _diag_block(arr_ellphi_bsbS, iband, istokes):
    """Extract diagonal (b,s; b,s) block -> (ℓ, φ)."""
    return np.real(arr_ellphi_bsbS[:, :, iband, istokes, iband, istokes])


def plot_cmb_pca_covariance(fitter, save=True, output_dir=None, nbins=50):
    """
    Plot CMB PCA covariance analysis showing raw, CMB, residual, and total covariance components.

    Parameters:
    ----------
    fitter : PolarizedBeamFitter or BootstrapBeamFitter
        The fitted beam fitter object
    save : bool
        Whether to save the plot
    output_dir : str, optional
        Output directory for plots. If None, uses fitter's config.
    nbins : int
        Number of radial bins for ell binning

    Returns:
    --------
    str or None
        Filename if saved, None otherwise
    """
    print("\n--- Generating CMB PCA Covariance Analysis ---")

    # Check if noise_bundle debug data is available
    if not hasattr(fitter, 'noise_bundle') or fitter.noise_bundle is None:
        print("Warning: No noise_bundle found in fitter. Skipping CMB PCA covariance plot.")
        return None

    if 'debug' not in fitter.noise_bundle:
        print("Warning: No 'debug' data in noise_bundle. Skipping CMB PCA covariance plot.")
        return None

    try:
        d = fitter.noise_bundle["debug"]

        # Extract required data
        cov_cmb = d["covariance_cmb"]
        cov_res = d["covariance_noncmb"]
        cov_total = d["covariance_total"]
        ell_x_grid = d["ell_x_grid"]
        ell_radial = d["ell_radial"]

    except KeyError as e:
        print(f"Warning: Missing required data in noise_bundle debug: {e}. Skipping CMB PCA covariance plot.")
        return None

    # Determine ellmax from grid if provided, else from ell_radial
    if ell_x_grid is not None:
        ellmax = float(np.nanmax(ell_x_grid))
    else:
        ellmax = float(np.nanmax(ell_radial))

    # Field means (over src if present)
    cmb_mean = _field_mean(cov_cmb)      # handles both with/without src axis
    res_mean = _field_mean(cov_res)
    tot_mean = _field_mean(cov_total)

    stokes_list = ["T", "Q", "U"]
    bands = list(fitter.config.bands)  # assume length 3, e.g. ["90GHz","150GHz","220GHz"]

    fig, axes = plt.subplots(
        nrows=3, ncols=3, figsize=(12, 10), sharex=True, sharey=True
    )

    # For a clean, consistent legend label ordering
    curve_order = [
        ("CMB model", cmb_mean),
        ("Residual model", res_mean),
        ("Total model", tot_mean),
    ]

    for iband, band in enumerate(bands):
        for istokes, stokes in enumerate(stokes_list):
            ax = axes[istokes, iband]

            # Extract diagonal blocks (ℓ, φ) for each component
            curves = []
            for label, arr in curve_order:
                block = _diag_block(arr, iband, istokes)
                c_ell, c_val, c_cnt = bin_ell_radial(ell_radial, block, ellmax, nbins=nbins)
                # keep NaNs for empty bins; semilogy will skip them
                curves.append((label, c_ell, c_val, c_cnt))

            # Plot — semilogy, default styles, consistent label order
            for (label, c_ell, c_val, c_cnt) in curves:
                ax.semilogy(c_ell, c_val, label=label)

            ax.set_ylim((1e1,1e7))
            ax.set_xlim((0,10000))

            ax.set_title(f"{band} • {stokes}")
            if iband == 0:
                ax.set_ylabel("Variance per mode (mK$^2$)")
            if istokes == 2:
                ax.set_xlabel(r"$\ell$")
            ax.grid(True, which="both", alpha=0.2)

    # Shared legend (single, outside plot)
    # Grab handles/labels from the last axes that has them
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels)

    plt.tight_layout()

    if save:
        if output_dir is None:
            output_dir = fitter.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = os.path.join(output_dir, "cmb_pca_covariance.png")
        plt.savefig(plot_filename, dpi=200)
        plt.close(fig)
        print(f"Saved CMB PCA covariance plot to: {plot_filename}")
        return plot_filename
    else:
        plt.show()
        return None


class BeamPlotter:
    """
    Class for creating various plots and visualizations for beam analysis.
    """

    def __init__(self, fitter, output_dir=None):
        """
        Initialize the plotter.

        Parameters:
        ----------
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

        # The new fitter is always multi-band capable
        self.is_multiband = len(self.base_fitter.config.bands) > 1

        self.primary_band = self.fitter.config.bands[0]  # Use first band as primary

        os.makedirs(self.output_dir, exist_ok=True)

    def _get_beam_model(self, band=None):
        """Get the appropriate beam model for the given band."""
        if band is None:
            band = self.primary_band
        return self.base_fitter.beam_models[band]

    def _get_band_suffix(self, band=None):
        """Get the band suffix for filenames."""
        if band is None:
            band = self.primary_band
        return band.replace("GHz", "")

    def _apply_ell_cut_indices(self, array_2d):
        """Apply fitter's ell truncation indices to a 2D Fourier array if needed."""
        idx_y = getattr(self.base_fitter, "k_indices_y", None)
        idx_x = getattr(self.base_fitter, "k_indices_x", None)
        if idx_y is None or idx_x is None:
            return array_2d
        if array_2d.shape[0] != len(idx_y) or array_2d.shape[1] != len(idx_x):
            array_2d = np.take(array_2d, idx_y, axis=0)
            array_2d = np.take(array_2d, idx_x, axis=1)
        return array_2d

    def _compute_asd_with_ell_cut(self, map_2d):
        """Compute ASD matching the fitter's ell truncation."""
        fft_map = np.fft.fft2(map_2d)
        fft_map = self._apply_ell_cut_indices(fft_map)
        asd = np.abs(fft_map)
        return np.fft.fftshift(asd)

    def _get_fit_params_for_band(self, best_fit_params, band=None):
        """Extract fit parameters for a specific band from multi-band results."""
        if band is None:
            band = self.primary_band

        # Get band index
        band_idx = self.fitter.config.bands.index(band)

        # Extract beam parameters for this band
        beam_params = best_fit_params["beams"][band_idx]

        # Return parameters in expected format for beam model
        return beam_params

    def plot_template_projection_analysis(self, best_fit_params, skip_sources=None, save=True):
        """
        Plot template projection analysis using the brightest source as template for each band.

        This method uses the brightest source in each band as a template to project out of
        extended/problematic sources to analyze their residual structure.

        Parameters:
        ----------
        best_fit_params : dict
            Best-fit parameters from optimization
        skip_sources : list, optional
            List of source names to analyze. If None, uses config.skip_sources
        save : bool
            Whether to save the plot

        Returns:
        --------
        list
            List of filenames if saved, empty list otherwise
        """
        if skip_sources is None:
            skip_sources = self.base_fitter.config.skip_sources

        if not skip_sources:
            print("No skip_sources specified for template projection analysis")
            return []

        maps_numpy = np.array(self.base_fitter.maps_jax)
        source_ids = self.base_fitter.source_ids
        flux = best_fit_params["sources"]["flux"]

        filenames = []
        for band_idx, band in enumerate(self.fitter.config.bands):
            band_suffix = self._get_band_suffix(band)
            print(f"\n--- Template Projection Analysis for {band} ---")

            # Calculate T amplitudes for the current band
            t_amps = np.array(flux[:, band_idx, 0])

            # Find brightest source as template *in this band*
            brightest_idx = np.argmax(t_amps)
            brightest_source_id = source_ids[brightest_idx]
            template_map = maps_numpy[brightest_idx, :, :, band_idx, 0]

            print(f"Using brightest source as template for {band}: {brightest_source_id}")
            print(f"Template amplitude: {t_amps[brightest_idx]:.1f} μK")

            # Find skip_sources in the data
            skip_sources_data = []
            for source_short_name in skip_sources:
                found = False
                for idx, full_source_id in enumerate(source_ids):
                    if source_short_name in full_source_id:
                        skip_sources_data.append((full_source_id, idx, source_short_name))
                        print(f"Found skip source for template analysis: {full_source_id}")
                        found = True
                        break
                if not found:
                    print(f"Warning: Could not find skip source '{source_short_name}' in data.")

            if not skip_sources_data:
                print(f"No specified skip sources found in the data for band {band}.")
                continue

            n_skip_sources = len(skip_sources_data)
            fig, axes = plt.subplots(n_skip_sources, 2, figsize=(12, 4 * n_skip_sources), squeeze=False)
            fig.suptitle(
                f"Template Projection Analysis ({band_suffix})\nTemplate: {brightest_source_id}",
                fontsize=14,
            )

            for i, (source_id, source_idx, short_name) in enumerate(skip_sources_data):
                data_map = maps_numpy[source_idx, :, :, band_idx, 0]
                template_flat = template_map.flatten()
                data_flat = data_map.flatten()
                alpha = np.dot(template_flat, data_flat) / np.dot(template_flat, template_flat)
                residual_map = data_map - alpha * template_map

                # Plot data
                ax = axes[i, 0]
                data_max = np.max(np.abs(data_map))
                im1 = ax.imshow(data_map, cmap="RdBu_r", origin="lower", vmin=-data_max, vmax=data_max)
                ax.set_title(f"{short_name}\nData T Map", fontsize=10)
                plt.colorbar(im1, ax=ax, label="T (μK)", fraction=0.046, pad=0.04)

                # Plot residual
                ax = axes[i, 1]
                residual_max = np.max(np.abs(residual_map))
                im2 = ax.imshow(residual_map, cmap="RdBu_r", origin="lower", vmin=-residual_max, vmax=residual_max)
                ax.set_title(f"Residual (α = {alpha:.3f})", fontsize=10)
                plt.colorbar(im2, ax=ax, label="Residual (μK)", fraction=0.046, pad=0.04)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            if save:
                plot_filename = os.path.join(self.output_dir, f"template_projection_analysis_{band_suffix}.png")
                plt.savefig(plot_filename, dpi=300)
                plt.close(fig)
                print(f"Saved template projection analysis for {band} to: {plot_filename}")
                filenames.append(plot_filename)
            else:
                plt.show()

        return filenames if save else None

    def plot_beam_profiles(self, best_fit_params, save=True, band=None):
        """
        Plot radial beam profiles and T-P beam difference with optional bootstrap uncertainties.

        Parameters:
        ----------
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
        model_type = self.base_fitter.config.beam_model_type.replace("_", "-").title()
        fig.suptitle(f"Best-Fit {model_type} Beam Model ({band_suffix})", fontsize=16)

        # Top panel: Final Beam profiles
        t_label = info.get("t_label", "Best-Fit T-Beam Profile")
        p_label = info.get("p_label", "Best-Fit P-Beam Profile")
        ylabel = info.get("ylabel", "Beam Amplitude")

        ax1.plot(r_fine, profile_T_fine, label=t_label, lw=3, color="C0", zorder=10)
        ax1.plot(
            r_fine,
            profile_P_fine,
            label=p_label,
            lw=3,
            linestyle="--",
            color="C1",
            zorder=10,
        )
        ax1.axhline(0, color="black", lw=0.5, zorder=1)

        y_min = min(np.min(profile_T_fine), np.min(profile_P_fine))
        y_max = max(np.max(profile_T_fine), np.max(profile_P_fine))
        y_range = y_max - y_min
        ax1.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        ax1.grid(True, which="both", linestyle=":", alpha=0.5)
        ax1.set_ylabel(ylabel, fontsize=12)
        ax1.set_title("Reconstructed Beam Profiles", fontsize=14)
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
        fig, axes = plt.subplots(
            len(self.fitter.config.bands),
            2,
            figsize=(14, 4 * len(self.fitter.config.bands)),
            sharex=True,
        )
        if len(self.fitter.config.bands) == 1:
            axes = axes.reshape(1, -1)

        model_type = self.base_fitter.config.beam_model_type.replace("_", "-").title()
        fig.suptitle(f"Multi-Band {model_type} Beam Models", fontsize=16)

        colors = ["C0", "C1", "C2"]

        for i, band in enumerate(self.fitter.config.bands):
            # Get fit parameters for this band
            fit_params = self._get_fit_params_for_band(best_fit_params, band)
            beam_model = self._get_beam_model(band)
            band_suffix = self._get_band_suffix(band)

            # Get beam profiles
            r_fine, profile_T_fine, profile_P_fine, info = beam_model.get_profiles_for_plotting(fit_params)

            # Top panel: Beam profiles
            ax = axes[i, 0]
            t_label = info.get("t_label", f"T-Beam ({band_suffix})")
            p_label = info.get("p_label", f"P-Beam ({band_suffix})")
            ax.plot(
                r_fine,
                profile_T_fine,
                label=t_label,
                lw=2,
                color=colors[i],
            )
            ax.plot(
                r_fine,
                profile_P_fine,
                label=p_label,
                lw=2,
                linestyle="--",
                color=colors[i],
                alpha=0.7,
            )
            ax.axhline(0, color="black", lw=0.5)
            ax.set_ylabel("Beam Amplitude", fontsize=12)
            ax.set_title(f"{band_suffix} Beam Profiles", fontsize=14)
            ax.grid(True, which="both", linestyle=":", alpha=0.5)
            ax.legend(fontsize=10)

            # Bottom panel: T-P difference
            ax = axes[i, 1]
            beam_difference = profile_T_fine - profile_P_fine
            ax.plot(
                r_fine,
                beam_difference,
                lw=2,
                color=colors[i],
                label=f"T-P ({band_suffix})",
            )
            ax.axhline(0, color="black", lw=0.5)
            ax.set_ylabel("Amplitude Difference", fontsize=12)
            ax.set_title(f"{band_suffix} T-Beam minus P-Beam", fontsize=14)
            ax.set_xlim(-0.2, 10.0)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(fontsize=10)

            if i == len(self.fitter.config.bands) - 1:  # bottom row
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
        ----------
        save : bool
            Whether to save the plot
        band : str, optional
            For multi-band fitters, specify which band to plot. If None, plots primary band.

        Returns:
        --------
        str or None
            Filename if saved, None otherwise
        """
        if self.base_fitter.config.beam_model_type != "bsplines":
            print(f"Skipping basis diagnostics for {self.base_fitter.config.beam_model_type} beam model")
            return None

        if band is None:
            band = self.primary_band

        beam_model = self._get_beam_model(band)
        band_suffix = self._get_band_suffix(band)
        print(f"\n--- Generating Basis Diagnostics for {band} ---")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Orthonormal Basis Diagnostics ({band_suffix})", fontsize=16)

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
        example_coeffs = [
            ([1, 0, 0, 0, 0], "Mode 0 only"),
            ([0, 1, 0, 0, 0], "Mode 1 only"),
            ([0.5, 0.3, 0.1, 0, 0], "Mixed modes"),
        ]

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
                gram_visual[i, j] = np.trapezoid(ortho_funcs[:, i] * ortho_funcs[:, j] * weight, r)

        im = ax.imshow(gram_visual, cmap="RdBu_r", vmin=-0.1, vmax=1.1)
        ax.set_title("Gram Matrix of Basis Functions")
        ax.set_xlabel("Basis function index")
        ax.set_ylabel("Basis function index")
        plt.colorbar(im, ax=ax)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save:
            plot_filename = os.path.join(self.output_dir, f"orthonormal_basis_{band_suffix}.png")
            plt.savefig(plot_filename, dpi=150)
            plt.close(fig)
            print(f"Saved basis diagnostics plot for {band} to: {plot_filename}")
            return plot_filename
        else:
            plt.show()
            return None

    def plot_source_diagnostics(self, best_fit_params, n_sources=None, central_crop=None, save=True):
        """
        Plot data/model/residual maps for the sources with highest polarization amplitude.

        Parameters:
        ----------
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
        if n_sources is None:
            n_sources = self.base_fitter.config.n_diagnostic_plots

        if n_sources == 0:
            print("Skipping source diagnostic plots (n_diagnostic_plots = 0)")
            return []

        total_sources = len(self.base_fitter.source_ids)
        if isinstance(n_sources, str) and n_sources.lower() == "all":
            n_to_plot = total_sources
            print(f"\n--- Generating Data/Model/Residual Maps for All {n_to_plot} Sources ---")
        else:
            n_to_plot = min(int(n_sources), total_sources)
            print(f"\n--- Generating Data/Model/Residual Maps for Top {n_to_plot} Highest P-amp Sources ---")

        if central_crop is not None:
            print(f"Using central {central_crop}x{central_crop} pixel crop")

        maps_numpy = np.array(self.base_fitter.maps_jax)
        model_maps = self.base_fitter.create_model_maps(best_fit_params)
        source_ids = self.base_fitter.source_ids
        flux = best_fit_params["sources"]["flux"]

        # Rank sources by polarization amplitude in the primary band
        primary_band_idx = self.fitter.config.bands.index(self.primary_band)
        p_amp_sources = []
        for i, source_id in enumerate(source_ids):
            p_amp = float(np.sqrt(flux[i, primary_band_idx, 1] ** 2 + flux[i, primary_band_idx, 2] ** 2))
            p_amp_sources.append((p_amp, source_id, i))
        p_amp_sources.sort(key=lambda x: x[0], reverse=True)

        print(f"Ranking sources by P-amplitude in primary band ({self.primary_band})")

        filenames = []
        for rank in range(n_to_plot):
            if rank < len(p_amp_sources):
                p_amp, source_id, idx = p_amp_sources[rank]
                print(f"\nCreating diagnostic plots for rank #{rank + 1} source: {source_id} (P-amp={p_amp:.1f} μK)")

                for band_idx, band in enumerate(self.fitter.config.bands):
                    filename = self._create_source_diagnostic_plot(
                        source_id, idx, rank + 1, maps_numpy, model_maps, band, band_idx, central_crop, save
                    )
                    if filename:
                        filenames.append(filename)
                        print(f"  Saved plot for band {band} to: {filename}")
        return filenames

    def _create_source_diagnostic_plot(
        self,
        source_id,
        source_idx,
        rank,
        data_maps,
        model_maps,
        band,
        band_idx,
        central_crop=None,
        save=True,
    ):
        """Create a diagnostic plot for a single source and band."""
        band_suffix = self._get_band_suffix(band)

        # Extract maps for the specified band
        data_T = data_maps[source_idx, :, :, band_idx, 0]
        data_Q = data_maps[source_idx, :, :, band_idx, 1]
        data_U = data_maps[source_idx, :, :, band_idx, 2]

        model_T = model_maps[source_idx, :, :, band_idx, 0]
        model_Q = model_maps[source_idx, :, :, band_idx, 1]
        model_U = model_maps[source_idx, :, :, band_idx, 2]

        residual_T = data_T - model_T
        residual_Q = data_Q - model_Q
        residual_U = data_U - model_U

        if central_crop:
            ny, nx = data_T.shape
            y_start = (ny - central_crop) // 2
            y_end = y_start + central_crop
            x_start = (nx - central_crop) // 2
            x_end = x_start + central_crop
            data_T, data_Q, data_U = (
                data_T[y_start:y_end, x_start:x_end],
                data_Q[y_start:y_end, x_start:x_end],
                data_U[y_start:y_end, x_start:x_end],
            )
            model_T, model_Q, model_U = (
                model_T[y_start:y_end, x_start:x_end],
                model_Q[y_start:y_end, x_start:x_end],
                model_U[y_start:y_end, x_start:x_end],
            )
            residual_T, residual_Q, residual_U = (
                residual_T[y_start:y_end, x_start:x_end],
                residual_Q[y_start:y_end, x_start:x_end],
                residual_U[y_start:y_end, x_start:x_end],
            )

        fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)
        crop_suffix = f" (Central {central_crop}x{central_crop})" if central_crop else ""
        fig.suptitle(f"Data/Model/Residual Maps for Source #{rank}: {source_id}{crop_suffix} ({band_suffix})", fontsize=16)

        maps_to_plot = {
            "T": (data_T, model_T, residual_T),
            "Q": (data_Q, model_Q, residual_Q),
            "U": (data_U, model_U, residual_U),
        }

        for i, stokes in enumerate(["T", "Q", "U"]):
            d_map, m_map, r_map = maps_to_plot[stokes]
            vmax = np.max(np.abs(d_map))
            res_vmax = np.max(np.abs(r_map))

            # Data, Model, Residual
            im_d = axes[i, 0].imshow(d_map, cmap="viridis", vmin=-vmax, vmax=vmax)
            im_m = axes[i, 1].imshow(m_map, cmap="viridis", vmin=-vmax, vmax=vmax)
            im_r = axes[i, 2].imshow(r_map, cmap="RdBu_r", vmin=-res_vmax, vmax=res_vmax)

            fig.colorbar(im_d, ax=axes[i, 0], label="μK")
            fig.colorbar(im_m, ax=axes[i, 1], label="μK")
            fig.colorbar(im_r, ax=axes[i, 2], label="μK")

            if i == 0:
                axes[i, 0].set_title("Data")
                axes[i, 1].set_title("Model")
                axes[i, 2].set_title("Residual")

            axes[i, 0].set_ylabel(stokes, fontsize=14, rotation=0, labelpad=20)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

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
        Plot 2D amplitude spectral density analysis for the top source in each band.

        Parameters:
        ----------
        best_fit_params : dict
            Best-fit parameters from optimization
        save : bool
            Whether to save the plot

        Returns:
        --------
        list
            List of filenames if saved, empty list otherwise
        """
        print("\n--- Generating ASD Analysis ---")
        maps_numpy = np.array(self.base_fitter.maps_jax)
        model_maps = self.base_fitter.create_model_maps(best_fit_params)
        flux = best_fit_params["sources"]["flux"]
        source_ids = self.base_fitter.source_ids

        filenames = []
        for band_idx, band in enumerate(self.fitter.config.bands):
            band_suffix = self._get_band_suffix(band)
            print(f"\n--- ASD Analysis for {band_suffix} ---")

            t_amps_band = np.array(flux[:, band_idx, 0])
            top_source_idx = np.argmax(t_amps_band)
            top_source_id = source_ids[top_source_idx]
            t_amp = t_amps_band[top_source_idx]

            print(f"Analyzing top T-amp source for {band}: {top_source_id} (T-amp = {t_amp:.1f} μK)")

            filename = self._create_asd_plot_for_source(top_source_id, top_source_idx, band, band_idx, maps_numpy, model_maps, save)
            if filename:
                filenames.append(filename)

        return filenames if save else None

    def _create_asd_plot_for_source(self, source_id, source_idx, band, band_idx, data_maps, model_maps, save):
        """Helper to create a single ASD plot for a given source and band."""
        if self.base_fitter.precision_jax is None:
            print("ASD analysis requires Fourier precision; skipping.")
            return None

        band_suffix = self._get_band_suffix(band)

        data_T, data_Q, data_U = [data_maps[source_idx, :, :, band_idx, i] for i in range(3)]
        model_T, model_Q, model_U = [model_maps[source_idx, :, :, band_idx, i] for i in range(3)]
        residual_T, residual_Q, residual_U = data_T - model_T, data_Q - model_Q, data_U - model_U

        fig, axes = plt.subplots(3, 4, figsize=(20, 12))
        fig.suptitle(f"2D Amplitude Spectral Density for {source_id} ({band_suffix})", fontsize=16)

        stokes_data = {
            "T": (data_T, model_T, residual_T),
            "Q": (data_Q, model_Q, residual_Q),
            "U": (data_U, model_U, residual_U),
        }

        for i, stokes in enumerate(["T", "Q", "U"]):
            d_map, m_map, r_map = stokes_data[stokes]
            asd_data = self._compute_asd_with_ell_cut(d_map)
            asd_model = self._compute_asd_with_ell_cut(m_map)
            asd_residual = self._compute_asd_with_ell_cut(r_map)

            precision_all = np.array(self.base_fitter.precision_jax)
            if precision_all.ndim == 5:
                precision = precision_all[source_idx, :, :, band_idx, i]
            elif precision_all.ndim == 7:
                precision = precision_all[source_idx, :, :, band_idx, i, band_idx, i]
            else:
                raise ValueError(f"Unexpected precision array shape for ASD plotting: {precision_all.shape}")

            precision = np.asarray(precision)
            if np.iscomplexobj(precision):
                precision = np.real(precision)

            with np.errstate(divide="ignore", invalid="ignore"):
                noise_psd = np.reciprocal(precision)

            finite_mask = np.isfinite(noise_psd) & (noise_psd > 0)
            noise_psd = np.where(finite_mask, noise_psd, np.inf)
            noise_psd_shifted = np.fft.fftshift(noise_psd)
            noise_asd = np.sqrt(noise_psd_shifted)
            asd_ratio = asd_residual / noise_asd

            # Plotting ASDs
            vmax = np.max(np.log10(asd_data + 1e-20))
            vmin = np.min(np.log10(asd_data + 1e-20))

            im_d = axes[i, 0].imshow(np.log10(asd_data + 1e-20), cmap="viridis", vmin=vmin, vmax=vmax)
            im_m = axes[i, 1].imshow(np.log10(asd_model + 1e-20), cmap="viridis", vmin=vmin, vmax=vmax)
            im_r = axes[i, 2].imshow(np.log10(asd_residual + 1e-20), cmap="viridis")
            log_asd_ratio = np.log10(asd_ratio + 1e-20)
            vmin = 0.9 * np.min(log_asd_ratio)
            vmax = 1.1 * np.max(log_asd_ratio)
            im_ratio = axes[i, 3].imshow(log_asd_ratio, cmap="viridis", vmin=vmin, vmax=vmax)

            fig.colorbar(im_d, ax=axes[i, 0], label="log10(ASD)")
            fig.colorbar(im_m, ax=axes[i, 1], label="log10(ASD)")
            fig.colorbar(im_r, ax=axes[i, 2], label="log10(ASD)")
            fig.colorbar(im_ratio, ax=axes[i, 3], label="log10(Res/Noise)")

            if i == 0:
                axes[i, 0].set_title("Data")
                axes[i, 1].set_title("Model")
                axes[i, 2].set_title("Residual")
                axes[i, 3].set_title("Residual / Noise")

            axes[i, 0].set_ylabel(stokes, fontsize=14, rotation=0, labelpad=20)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save:
            safe_source_id = safe_filename(source_id)
            plot_filename = os.path.join(self.output_dir, f"asd_2d_{band_suffix}_{safe_source_id}.png")
            plt.savefig(plot_filename, dpi=150)
            plt.close(fig)
            print(f"Saved 2D ASD analysis plot to: {plot_filename}")
            return plot_filename
        else:
            plt.show()
            return None

    def _process_sampling_output(self, sampling_output, sampler_type="nuts"):
        """
        Process sampling output for plotting by reshaping chains and flattening structure.
        Supports both NUTS and MCLMC output formats.

        Returns:
        --------
        tuple
            (az_data, samples_flat, n_chains) for use in plotting methods
        """
        if sampler_type == "nuts":
            mcmc = sampling_output.get("mcmc")
            if mcmc is None:
                raise ValueError("Expected NUTS output to include an 'mcmc' object with metadata")
            n_chains = int(getattr(mcmc, "num_chains", 1))
        elif sampler_type == "mclmc":
            # Default to a single chain unless the caller provided explicit metadata
            n_chains = int(sampling_output.get("num_chains", 1))
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")

        samples_phys = sampling_output["samples_phys"]

        # Helper: reshape leading axis into (chain, draw, …)
        def _reshape_chain_draw(arr, n_chains):
            arr = np.asarray(arr)

            if arr.ndim >= 2 and arr.shape[0] == n_chains:
                return arr

            total_draws = arr.shape[0]
            if n_chains <= 0:
                raise ValueError(f"Invalid number of chains: {n_chains}")

            n_draws, remainder = divmod(total_draws, n_chains)
            if remainder != 0:
                raise ValueError(
                    f"Cannot reshape samples into {n_chains} chains: leading axis {total_draws} is not divisible by {n_chains}"
                )

            new_shape = (n_chains, n_draws) + arr.shape[1:]
            return arr.reshape(new_shape)

        # Helper: flatten nested dict for ArviZ
        def _flatten_for_arviz(samples_phys):
            flat = {}
            # beams
            for b_idx, beam_dict in enumerate(samples_phys["beams"]):
                for name, arr in beam_dict.items():
                    flat[f"beam_{b_idx}_{name}"] = _reshape_chain_draw(arr, n_chains)
            # sources
            flat["yoff"] = _reshape_chain_draw(samples_phys["sources"]["yoff"], n_chains)
            flat["xoff"] = _reshape_chain_draw(samples_phys["sources"]["xoff"], n_chains)
            flat["flux"] = _reshape_chain_draw(samples_phys["sources"]["flux"], n_chains)
            return flat

        samples_flat = _flatten_for_arviz(samples_phys)
        az_data = az.from_dict(samples_flat)

        return az_data, samples_flat, n_chains

    def plot_sampling_trace(self, sampling_output, sampler_type="nuts", save=True):
        """
        Plot trace plots for sampling output (NUTS or MCLMC).
        """
        sampler_name = sampler_type.upper()
        print(f"\n--- Generating {sampler_name} Trace Plots ---")

        az_data, samples_flat, n_chains = self._process_sampling_output(sampling_output, sampler_type)
        active_model_bounds = self.fitter.config.active_beam_model_bounds

        # Plot beam parameters
        for band_idx, band in enumerate(self.fitter.config.bands):
            band_suffix = self._get_band_suffix(band)
            beam_param_names = list(active_model_bounds.keys())
            beam_params = [f"beam_{band_idx}_{p}" for p in beam_param_names]
            az.plot_trace(az_data, var_names=beam_params)
            plt.suptitle(f"{sampler_name} Trace for Beam Parameters ({band_suffix})", y=1.02)
            if save:
                filename = os.path.join(self.output_dir, f"{sampler_type}_trace_beam_{band_suffix}.png")
                plt.tight_layout()
                plt.savefig(filename, dpi=200)
                plt.close()
                print(f"Saved beam trace plot to: {filename}")

        # Plot source parameters
        stokes_names = ["T", "Q", "U"]
        for band_idx, band in enumerate(self.fitter.config.bands):
            band_suffix = self._get_band_suffix(band)
            for stokes_idx, stokes in enumerate(stokes_names):
                var_name = "flux"
                coords = {"flux_dim_1": band_idx, "flux_dim_2": stokes_idx}
                az.plot_trace(az_data, var_names=[var_name], coords=coords)
                plt.suptitle(f"{sampler_name} Trace for Flux ({stokes}, {band_suffix})", y=1.02)
                if save:
                    filename = os.path.join(self.output_dir, f"{sampler_type}_trace_flux_{stokes}_{band_suffix}.png")
                    plt.tight_layout()
                    plt.savefig(filename, dpi=200)
                    plt.close()
                    print(f"Saved flux trace plot to: {filename}")

        # Plot offsets
        az.plot_trace(az_data, var_names=["yoff", "xoff"])
        plt.suptitle(f"{sampler_name} Trace for Source Offsets", y=1.02)
        if save:
            filename = os.path.join(self.output_dir, f"{sampler_type}_trace_offsets.png")
            plt.tight_layout()
            plt.savefig(filename, dpi=200)
            plt.close()
            print(f"Saved offset trace plot to: {filename}")

    def plot_nuts_trace(self, nuts_output, save=True):
        """
        Plot trace plots for NUTS samples.
        """
        return self.plot_sampling_trace(nuts_output, sampler_type="nuts", save=save)

    def plot_mclmc_trace(self, mclmc_output, save=True):
        """
        Plot trace plots for MCLMC samples.
        """
        return self.plot_sampling_trace(mclmc_output, sampler_type="mclmc", save=save)

    def plot_sampling_corner(self, sampling_output, sampler_type="nuts", chain_descriptions=None):
        """
        Corner plot for sampling output (NUTS or MCLMC).
        Shows ⟨T⟩, ⟨P⟩, ⟨y_off⟩, ⟨x_off⟩ plus all beam parameters.

        Parameters:
        -----------
        sampling_output : dict or list of dict
            Single sampling output dict, or list of sampling output dicts for comparison
        sampler_type : str
            Sampler type ("nuts" or "mclmc")
        chain_descriptions : list of str, optional
            Text descriptions for each chain (only used when sampling_output is a list)
        """
        # Handle single output vs list of outputs
        if not isinstance(sampling_output, list):
            sampling_output = [sampling_output]
            is_single_chain = True
        else:
            is_single_chain = False

        if chain_descriptions is None:
            chain_descriptions = [f"Chain {i + 1}" for i in range(len(sampling_output))]
        elif len(chain_descriptions) != len(sampling_output):
            raise ValueError(
                f"Number of chain_descriptions ({len(chain_descriptions)}) must match number of chains ({len(sampling_output)})"
            )

        sampler_name = sampler_type.upper()
        if is_single_chain:
            print(f"\n--- Generating {sampler_name} Corner Plot ---")
        else:
            print(f"\n--- Generating {sampler_name} Comparison Corner Plot for {len(sampling_output)} Chains ---")

        # Process all chains
        all_corner_arrays = []
        all_labels = None
        colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]  # Default matplotlib colors
        active_model_bounds = self.fitter.config.active_beam_model_bounds

        for i, output in enumerate(sampling_output):
            az_data, samples_flat, _ = self._process_sampling_output(output, sampler_type)

            def _merge_cd(arr):
                c, d = arr.shape[:2]
                return arr.reshape(c * d, *arr.shape[2:])  # (nsamp, ...)

            flux_flat = _merge_cd(samples_flat["flux"])  # (nsamp, n_src, n_bands, 3)
            primary_band = 0

            t_flux = flux_flat[:, :, primary_band, 0].mean(axis=1)  # <T> per sample
            p_flux = np.sqrt(flux_flat[:, :, primary_band, 1] ** 2 + flux_flat[:, :, primary_band, 2] ** 2).mean(axis=1)  # <P> per sample

            yoff_mean = _merge_cd(samples_flat["yoff"]).mean(axis=1)
            xoff_mean = _merge_cd(samples_flat["xoff"]).mean(axis=1)

            label_map = {
                "t_flux": r"$\langle T \rangle$",
                "p_flux": r"$\langle P \rangle$",
                "yoff_mean": r"$\langle y_{\rm off} \rangle$",
                "xoff_mean": r"$\langle x_{\rm off} \rangle$",
            }
            corner_data = {
                "t_flux": t_flux,
                "p_flux": p_flux,
                "yoff_mean": yoff_mean,
                "xoff_mean": xoff_mean,
            }

            for band_idx, band in enumerate(self.fitter.config.bands):
                band_suffix = self._get_band_suffix(band)
                for param_name in active_model_bounds.keys():
                    key_flat = f"beam_{band_idx}_{param_name}"
                    var_flat = _merge_cd(samples_flat[key_flat])  # (nsamp,)
                    if "beta" in param_name:
                        role = "T" if "T" in param_name else r"\mathrm{pol}"
                        pretty = rf"$\beta_{{{role}, {band_suffix}}}$"
                    else:
                        pretty = f"{param_name.replace('_', ' ')} ({band_suffix})"
                    label_map[key_flat] = pretty
                    corner_data[key_flat] = var_flat

            param_order = list(corner_data.keys())  # deterministic order
            corner_array = np.column_stack([corner_data[k] for k in param_order])

            # Store labels from first chain
            if all_labels is None:
                all_labels = [label_map.get(k, k) for k in param_order]

            all_corner_arrays.append(corner_array)

        # Create the corner plot
        if is_single_chain:
            # Single chain - original behavior
            fig = corner.corner(
                all_corner_arrays[0],
                labels=all_labels,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 10},
            )
            fig.suptitle(f"{sampler_name} Corner Plot", y=1.02)
        else:
            # Multiple chains - overlaid comparison
            fig = corner.corner(
                all_corner_arrays[0],
                labels=all_labels,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                color=colors[0],
                hist_kwargs={"alpha": 0.6},
                contour_kwargs={"alpha": 0.6},
                title_kwargs={"fontsize": 10},
            )

            # Overlay additional chains
            for i in range(1, len(all_corner_arrays)):
                fig = corner.corner(
                    all_corner_arrays[i],
                    fig=fig,
                    color=colors[i % len(colors)],
                    hist_kwargs={"alpha": 0.6},
                    contour_kwargs={"alpha": 0.6},
                    title_kwargs={"fontsize": 10},
                )

            # Add legend in upper right corner
            legend_elements = [
                plt.Line2D([0], [0], color=colors[i % len(colors)], lw=2, label=desc) for i, desc in enumerate(chain_descriptions)
            ]

            # Find the top-right unused corner and add legend
            ndim = len(all_labels)
            ax_legend = fig.axes[ndim - 1]  # Top-right subplot
            ax_legend.legend(handles=legend_elements, loc="center", frameon=True, fancybox=True, shadow=True, fontsize=10)

            fig.suptitle(f"{sampler_name} Comparison Corner Plot", y=1.02)

        filename = os.path.join(self.output_dir, f"{sampler_type}_corner{'_comparison' if not is_single_chain else ''}.png")
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()

        if is_single_chain:
            print(f"Saved corner plot to: {filename}")
        else:
            print(f"Saved comparison corner plot to: {filename}")

    def plot_nuts_corner(self, nuts_output, chain_descriptions=None):
        """
        Corner plot for NUTS samples.

        Parameters:
        -----------
        nuts_output : dict or list of dict
            Single NUTS output dict, or list of NUTS output dicts for comparison
        chain_descriptions : list of str, optional
            Text descriptions for each chain (only used when nuts_output is a list)
        """
        return self.plot_sampling_corner(nuts_output, sampler_type="nuts", chain_descriptions=chain_descriptions)

    def plot_mclmc_corner(self, mclmc_output, chain_descriptions=None):
        """
        Corner plot for MCLMC samples.

        Parameters:
        -----------
        mclmc_output : dict or list of dict
            Single MCLMC output dict, or list of MCLMC output dicts for comparison
        chain_descriptions : list of str, optional
            Text descriptions for each chain (only used when mclmc_output is a list)
        """
        return self.plot_sampling_corner(mclmc_output, sampler_type="mclmc", chain_descriptions=chain_descriptions)

    def plot_cmb_pca_covariance(self, save=True, nbins=50):
        """
        Plot CMB PCA covariance analysis showing raw, CMB, residual, and total covariance components.

        Parameters:
        ----------
        save : bool
            Whether to save the plot
        nbins : int
            Number of radial bins for ell binning

        Returns:
        --------
        str or None
            Filename if saved, None otherwise
        """
        return plot_cmb_pca_covariance(self.fitter, save=save, output_dir=self.output_dir, nbins=nbins)


def create_diagnostic_plots(
    fitter,
    best_fit_params,
    output_dir=None,
    include_template_analysis=False,
    central_crop=None,
    include_cmb_pca_covariance=False,
):
    """
    Convenience function to create all diagnostic plots.

    Parameters:
    ----------
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
    include_cmb_pca_covariance : bool, optional
        Whether to include CMB PCA covariance analysis plot

    Returns:
    --------
    dict
        Dictionary of plot filenames
    """
    plotter = BeamPlotter(fitter, output_dir)
    filenames = {}

    # Beam profiles (already handles multi-band correctly)
    filenames["beam_profiles"] = plotter.plot_beam_profiles(best_fit_params)

    # Basis diagnostics (per band)
    if plotter.base_fitter.config.beam_model_type == "bsplines":
        filenames["basis_diagnostics"] = []
        for band in plotter.fitter.config.bands:
            filename = plotter.plot_basis_diagnostics(save=True, band=band)
            if filename:
                filenames["basis_diagnostics"].append(filename)

    # Source diagnostics (now creates plots for all bands for top sources)
    filenames["source_diagnostics"] = plotter.plot_source_diagnostics(best_fit_params, central_crop=central_crop)

    # ASD analysis (now creates plots for all bands)
    filenames["asd_analysis"] = plotter.plot_asd_analysis(best_fit_params)

    # Template projection analysis (optional, now for all bands)
    if include_template_analysis:
        filenames["template_projection"] = plotter.plot_template_projection_analysis(best_fit_params)

    # CMB PCA covariance analysis (optional)
    if include_cmb_pca_covariance:
        filename = plotter.plot_cmb_pca_covariance()
        if filename:
            filenames["cmb_pca_covariance"] = filename

    print("\n--- All plotting complete. ---")
    return filenames


def create_nuts_plots(fitter, nuts_output, output_dir=None):
    """
    Convenience function to create all NUTS diagnostic plots.
    """
    plotter = BeamPlotter(fitter, output_dir)
    plotter.plot_nuts_trace(nuts_output)
    plotter.plot_nuts_corner(nuts_output)
    print("\n--- All NUTS plotting complete. ---")


def create_mclmc_plots(fitter, mclmc_output, output_dir=None):
    """
    Convenience function to create all MCLMC diagnostic plots.
    """
    plotter = BeamPlotter(fitter, output_dir)
    plotter.plot_mclmc_trace(mclmc_output)
    plotter.plot_mclmc_corner(mclmc_output)
    print("\n--- All MCLMC plotting complete. ---")


def create_sampling_plots(fitter, sampling_output, sampler_type="nuts", output_dir=None):
    """
    Convenience function to create all sampling diagnostic plots.
    Supports both NUTS and MCLMC samplers.
    """
    plotter = BeamPlotter(fitter, output_dir)
    plotter.plot_sampling_trace(sampling_output, sampler_type)
    plotter.plot_sampling_corner(sampling_output, sampler_type)
    sampler_name = sampler_type.upper()
    print(f"\n--- All {sampler_name} plotting complete. ---")
