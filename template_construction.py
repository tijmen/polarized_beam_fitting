"""Construct shifted T->P leakage templates for multiple fields and bands.

This script allows command-line regeneration of the T->P leakage template maps.
It applies a radial low-pass filter using the TOD Nyquist estimate, recenters
residual maps with bilinear interpolation, and stores source offsets alongside
each template so they can be used as initial guesses in later optimization runs.

One important thing to note is that this uses the fitter, which itself uses
cached leakage-cleaned inputs. For iterative cleaning like we did for dH26,
the procedure is:

First iteration:
 - Clear cache directory
 - Set `use_precomputed_leakage_templates = False` at the bottom of this script
 - Run this script

Subsequent iterations:
 - Clear cache directory
 - Set `use_precomputed_leakage_templates = True` at the bottom of this script
 - Run this script

"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("/home/tijmen/cmb_analysis/beam_analysis")

from polarized_beam_fitting import PolarizedBeamFitter
from polarized_beam_fitting.beam_model import create_beam_model
from polarized_beam_fitting.config import BeamFittingConfig
from polarized_beam_fitting.utils import (
    apply_radial_lowpass,
    ell_grid,
    make_apodization_mask,
    parse_declination,
    predict_nyquist_ell_x,
    shift_map_bilinear,
)


def plot_templates(templates: Dict[str, Dict[str, np.ndarray]], title: str, filename: Path) -> None:
    """Save a diagnostic plot for the set of templates."""
    n_cols = len(templates)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8), sharex=True, sharey=True)
    fig.suptitle(title, fontsize=16, y=0.98)

    vmax = 0.02
    schemes = list(templates.keys())
    for col, scheme in enumerate(schemes):
        q_template = templates[scheme]["Q"]
        u_template = templates[scheme]["U"]

        ax_q = axes[0, col]
        ax_q.imshow(q_template[100:199, 100:199], cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax_q.set_title(f"{scheme} weighting")

        ax_u = axes[1, col]
        ax_u.imshow(u_template[100:199, 100:199], cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")

    axes[0, 0].set_ylabel("Q template", fontsize=14)
    axes[1, 0].set_ylabel("U template", fontsize=14)

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def safe_float(value) -> float:
    """Convert a (possibly JAX) array-like value to a float."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.reshape(-1)[0])


def prepare_frequency_grids(map_shape: Tuple[int, int], reso_arcmin: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Pre-compute |ell| grid and taper width for filtering."""
    ny, nx = map_shape
    reso_deg = reso_arcmin / 60.0

    # Get ell coordinates
    _, _, ell_y_grid, ell_x_grid = ell_grid(map_shape, reso_arcmin)
    ell_radius = np.sqrt(ell_x_grid**2 + ell_y_grid**2)

    # Compute frequency steps in k-space for taper width conversion
    ky_freq = np.fft.fftfreq(ny, d=reso_deg)
    kx_freq = np.fft.fftfreq(nx, d=reso_deg)
    freq_steps: Iterable[float] = []
    if ny > 1:
        freq_steps = [abs(ky_freq[1] - ky_freq[0])]
    if nx > 1:
        freq_steps = list(freq_steps) + [abs(kx_freq[1] - kx_freq[0])]

    taper_width_cpd = 0.0
    if freq_steps:
        taper_width_cpd = 5.0 * min(freq_steps)

    return ell_radius, reso_deg, taper_width_cpd


def construct_templates(config, make_plots):
    print("Building T->P leakage templates...")

    fitter = PolarizedBeamFitter(config)
    best_fit_params = fitter.run_fit()

    print(f"Fit completed. Beam parameters: {best_fit_params['beams']}")

    all_source_ids = np.asarray(fitter.source_ids)
    all_source_fields = np.asarray(fitter.fields)
    raw_maps_all = np.asarray(fitter.raw_maps_numpy)
    yoff_all = np.asarray(best_fit_params["sources"]["yoff"], dtype=float)
    xoff_all = np.asarray(best_fit_params["sources"]["xoff"], dtype=float)
    flux_all = np.asarray(best_fit_params["sources"]["flux"], dtype=float)

    expected_shape = (len(all_source_ids), len(config.bands))
    if yoff_all.shape != expected_shape or xoff_all.shape != expected_shape:
        raise ValueError(
            f"Unexpected source offset shape from joint fit: yoff={yoff_all.shape}, xoff={xoff_all.shape}, expected {expected_shape}."
        )
    if flux_all.shape != (len(all_source_ids), len(config.bands), 3):
        raise ValueError(
            f"Unexpected source flux shape from joint fit: flux={flux_all.shape}, expected {(len(all_source_ids), len(config.bands), 3)}."
        )

    map_shape = (config.map_size_pix, config.map_size_pix)
    y_coords = np.arange(-map_shape[0] // 2, map_shape[0] // 2)
    x_coords = np.arange(-map_shape[1] // 2, map_shape[1] // 2)
    y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing="ij")
    beam_models = {band: create_beam_model(config, y_grid, x_grid, band) for band in config.bands}

    apod_mask = make_apodization_mask(map_shape, config.apodization_width_pix)
    ell_radius, _, taper_width_cpd = prepare_frequency_grids(map_shape, config.reso_arcmin)
    taper_width_ell = 360.0 * taper_width_cpd

    for field in config.coadd_filenames.keys():
        field_source_indices = np.where(all_source_fields == field)[0]
        if field_source_indices.size == 0:
            raise ValueError(f"No fitted sources found for field '{field}'.")

        for band_idx, band in enumerate(config.bands):
            msg = f"= Building templates for field={field}, band={band} ... ="
            print((len(msg)) * "=")
            print(msg)
            print((len(msg)) * "=")

            beam_model = beam_models[band]
            band_beam_params = best_fit_params["beams"][band_idx]

            normalized_q_maps = []
            normalized_u_maps = []
            source_t_amps = []
            source_offsets: Dict[str, Dict[str, float]] = {}
            source_flux: Dict[str, Dict[str, np.ndarray]] = {}
            source_kmax: Dict[str, float | None] = {}

            for source_idx in field_source_indices:
                source_base = str(all_source_ids[source_idx])
                raw_maps = raw_maps_all[source_idx, :, :, band_idx, :]
                yoff = safe_float(yoff_all[source_idx, band_idx])
                xoff = safe_float(xoff_all[source_idx, band_idx])
                flux = np.asarray(flux_all[source_idx, band_idx, :], dtype=float)
                t_amp, q_amp, u_amp = flux

                if abs(t_amp) <= 1e-6:
                    print(f"  Warning: T-amplitude for {source_base} is too small ({t_amp}). Skipping source.")
                    continue

                _, p_beam_map = beam_model.evaluate_beam_maps(band_beam_params, yoff, xoff)
                p_beam_map = np.array(p_beam_map)

                q_model = q_amp * p_beam_map
                u_model = u_amp * p_beam_map

                q_residual = raw_maps[:, :, 1] - q_model
                u_residual = raw_maps[:, :, 2] - u_model

                declination = parse_declination(source_base)
                nyquist_ell_x = predict_nyquist_ell_x(declination)
                ellmax = 0.85 * nyquist_ell_x if np.isfinite(nyquist_ell_x) else np.inf
                source_kmax[source_base] = None if not np.isfinite(ellmax) else float(ellmax / 360.0)  # Keep as k for compatibility

                q_filtered = apply_radial_lowpass(q_residual, apod_mask, ell_radius, ellmax, taper_width_ell)
                u_filtered = apply_radial_lowpass(u_residual, apod_mask, ell_radius, ellmax, taper_width_ell)

                q_normalized = q_filtered / t_amp
                u_normalized = u_filtered / t_amp

                normalized_q_maps.append(shift_map_bilinear(q_normalized, -yoff, -xoff))
                normalized_u_maps.append(shift_map_bilinear(u_normalized, -yoff, -xoff))
                source_t_amps.append(t_amp)
                source_offsets[source_base] = {"y_offset": float(yoff), "x_offset": float(xoff)}
                per_source_flux = source_flux.setdefault(source_base, {})
                per_source_flux[band] = flux.copy()

            if len(normalized_q_maps) < 2:
                raise ValueError(f"Fewer than 2 sources available for field '{field}', band '{band}'. Cannot build a robust template.")

            q_stack = np.stack(normalized_q_maps, axis=0)
            u_stack = np.stack(normalized_u_maps, axis=0)
            t_amps = np.asarray(source_t_amps, dtype=float)

            templates: Dict[str, Dict[str, np.ndarray]] = {}
            templates["median"] = {"Q": np.median(q_stack, axis=0), "U": np.median(u_stack, axis=0)}
            templates["flat"] = {"Q": np.mean(q_stack, axis=0), "U": np.mean(u_stack, axis=0)}

            weights_lin = t_amps[:, np.newaxis, np.newaxis]
            templates["linear"] = {
                "Q": np.sum(q_stack * weights_lin, axis=0) / np.sum(weights_lin),
                "U": np.sum(u_stack * weights_lin, axis=0) / np.sum(weights_lin),
            }

            weights_sq = t_amps[:, np.newaxis, np.newaxis] ** 2
            templates["quadratic"] = {
                "Q": np.sum(q_stack * weights_sq, axis=0) / np.sum(weights_sq),
                "U": np.sum(u_stack * weights_sq, axis=0) / np.sum(weights_sq),
            }

            suffix = "_cdrc" if config.use_cdrc else ""
            plotting_title = f"Filtered Leakage Templates\n{field} - {band}{suffix}"
            if make_plots:
                plot_path = Path(config.leakage_template_dir) / f"templates_{field}_{band}{suffix}.png"
                plot_templates(templates, plotting_title, plot_path)

            metadata = {
                "coadd": field,
                "band": band,
                "apod_width_pix": config.apodization_width_pix,
                "taper_width_pixels": 5,
                "reso_arcmin": config.reso_arcmin,
                "use_cdrc": config.use_cdrc,
            }

            for scheme, scheme_maps in templates.items():
                payload = {
                    "Q": scheme_maps["Q"],
                    "U": scheme_maps["U"],
                    "source_offsets": {
                        sid: {"y_offset": float(offsets["y_offset"]), "x_offset": float(offsets["x_offset"])}
                        for sid, offsets in source_offsets.items()
                    },
                    "source_flux": {
                        sid: {band_key: {"T": float(vec[0]), "Q": float(vec[1]), "U": float(vec[2])} for band_key, vec in per_band.items()}
                        for sid, per_band in source_flux.items()
                    },
                    "source_kmax_cpd": {sid: (None if value is None else float(value)) for sid, value in source_kmax.items()},
                    "metadata": metadata,
                }

                output_path = Path(config.leakage_template_dir) / f"leakage_template_{field}_{band}_{scheme}{suffix}.pkl"
                with output_path.open("wb") as handle:
                    pickle.dump(payload, handle)
                print(f"  Saved {scheme} template to {output_path}")


if __name__ == "__main__":
    from polarized_beam_fitting import BeamFittingConfig

    config = BeamFittingConfig()

    # Important! Set this to False for the first run. True for subsequent iterations.
    config.use_precomputed_leakage_templates = True

    # example modifications to the config:
    # config.bands = ["90GHz"]
    # config.coadd_filenames = {"winter": ["/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_tau_decon_winter.g3"]}
    # config.use_cdrc = True
    # config.leakage_template_dir = "/home/tijmen/cmb_analysis/beam_analysis/leakage_templates"

    construct_templates(config, make_plots=True)
