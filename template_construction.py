"""Construct shifted T->P leakage templates for multiple fields and bands.

This script allows command-line regeneration of the T->P leakage template maps.
It applies a radial low-pass filter using the TOD Nyquist estimate, recenters
residual maps with bilinear interpolation, and stores source offsets alongside
each template so they can be used as initial guesses in later optimization runs.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple
from copy import copy

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("/home/tijmen/cmb_analysis/beam_analysis")

from polarized_beam_fitting import PolarizedBeamFitter
from polarized_beam_fitting.beam_model import create_beam_model
from polarized_beam_fitting.config import BeamFittingConfig
from polarized_beam_fitting.data_loader import load_g3_source_map_records
from polarized_beam_fitting.utils import (
    apply_radial_lowpass,
    ell_grid,
    make_apodization_mask,
    match_subfield_by_declination,
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


def _get_cdrc_params(config: BeamFittingConfig, source_id: str, band: str, field: str) -> Tuple[Dict[str, float], str]:
    """Return CDRC parameters and matched subfield name for a source."""
    if field not in ["winter", "winter_nodecon"]:
        raise ValueError(f"CDRC is only configured for the winter field; got field='{field}'.")
    subfield_name = match_subfield_by_declination(source_id, config.cdrc_winter_params.keys())
    band_params = config.cdrc_winter_params[subfield_name].get(band)
    if band_params is None:
        raise ValueError(f"No CDRC parameters found for band '{band}' in subfield '{subfield_name}'.")
    return band_params, subfield_name


def _build_cdrc_transform(config: BeamFittingConfig, band: str, params: Dict[str, float]) -> np.ndarray:
    """Build the 3x3 CDRC transform matrix for (T, Q, U)."""
    tcal = config.cmb_calibration_factors["T"][band]
    pcal = config.cmb_calibration_factors["Q"][band]
    eps_q = float(params["epsilon_q_tt"])
    eps_u = float(params["epsilon_u_tt"])
    psi = float(params["delta_psi"])

    c = np.cos(psi)
    s = np.sin(psi)
    pscale = pcal / tcal

    a1 = np.diag([tcal, tcal, tcal])
    a2 = np.array([[1.0, 0.0, 0.0], [-eps_q, 1.0, 0.0], [-eps_u, 0.0, 1.0]])
    a3 = np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])
    a4 = np.diag([1.0, pscale, pscale])
    return a4 @ a3 @ a2 @ a1


def load_raw_maps_for_band(config: BeamFittingConfig, fitter: PolarizedBeamFitter, field: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract raw T/Q/U maps from the G3 file for sources present in the fitter state."""
    raw_maps_data: Dict[str, Dict[str, np.ndarray]] = {}
    band = config.bands[0]
    source_ids = set(fitter.source_ids)

    filenames = config.coadd_file_list
    if not filenames:
        raise ValueError("No coadd files configured for template construction.")
    filename = filenames[0]
    try:
        records = load_g3_source_map_records(filename, [band], source_bases=source_ids)
        for record in records:
            source_base = record.source_id.split(f"-{band}")[0]
            t_map_np = record.t
            q_map_np = record.q
            u_map_np = record.u

            if config.use_cdrc:
                params, subfield_name = _get_cdrc_params(config, source_base, band, field)
                transform = _build_cdrc_transform(config, band, params)
                stacked = np.stack((t_map_np, q_map_np, u_map_np), axis=-1)
                stacked = np.einsum("wv,yxv->yxw", transform, stacked, optimize=True)
                t_map_np, q_map_np, u_map_np = (stacked[:, :, 0], stacked[:, :, 1], stacked[:, :, 2])
                print(f"Applied map-domain CDRC to {source_base} (field={field}, subfield={subfield_name}, band={band}).")

            raw_maps_data[source_base] = {"T": t_map_np, "Q": q_map_np, "U": u_map_np}
    except RuntimeError as err:
        raise RuntimeError(f"Could not open coadd file {filename}") from err

    if not raw_maps_data:
        raise ValueError(f"No sources found in {filename} for band {band}.")
    return raw_maps_data


def construct_templates(config, make_plots):
    """Loop over fields and bands to build T->P leakage templates."""

    fields_to_process = copy(config.coadd_filenames.keys())
    bands_to_process = copy(config.bands)

    # Templates are built for single-band, single-field processing
    for field in fields_to_process:
        for band in bands_to_process:
            config.coadd_filenames = {field: [config.coadd_filenames[field][0]]}
            config.bands = [band]
            msg = f"= Processing field={field}, band={band} ... ="
            print((len(msg)) * "=")
            print(msg)
            print((len(msg)) * "=")

            fitter = PolarizedBeamFitter(config)
            best_fit_params = fitter.run_fit()
            all_source_ids = np.asarray(fitter.source_ids)

            map_shape = (config.map_size_pix, config.map_size_pix)
            y_coords = np.arange(-map_shape[0] // 2, map_shape[0] // 2)
            x_coords = np.arange(-map_shape[1] // 2, map_shape[1] // 2)
            y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing="ij")
            beam_model = create_beam_model(config, y_grid, x_grid, band)
            band_beam_params = best_fit_params["beams"][0]

            apod_mask = make_apodization_mask(map_shape, config.apodization_width_pix)
            ell_radius, _, taper_width_cpd = prepare_frequency_grids(map_shape, config.reso_arcmin)
            raw_maps_data = load_raw_maps_for_band(config, fitter, field)

            normalized_q_maps = []
            normalized_u_maps = []
            source_t_amps = []
            source_offsets: Dict[str, Dict[str, float]] = {}
            source_flux: Dict[str, Dict[str, np.ndarray]] = {}
            source_kmax: Dict[str, float | None] = {}

            for source_base, raw_maps in raw_maps_data.items():
                match = np.where(all_source_ids == source_base)[0]
                if match.size == 0:
                    print(f"  Warning: Could not find '{source_base}' in best-fit results. Skipping source.")
                    continue
                source_idx = int(match[0])

                yoff = safe_float(best_fit_params["sources"]["yoff"][source_idx])
                xoff = safe_float(best_fit_params["sources"]["xoff"][source_idx])
                flux = np.asarray(best_fit_params["sources"]["flux"][source_idx, 0, :], dtype=float)
                t_amp, q_amp, u_amp = flux

                if abs(t_amp) <= 1e-6:
                    print(f"  Warning: T-amplitude for {source_base} is too small ({t_amp}). Skipping source.")
                    continue

                _, p_beam_map = beam_model.evaluate_beam_maps(band_beam_params, yoff, xoff)
                p_beam_map = np.array(p_beam_map)

                q_model = q_amp * p_beam_map
                u_model = u_amp * p_beam_map

                q_residual = raw_maps["Q"] - q_model
                u_residual = raw_maps["U"] - u_model

                declination = parse_declination(source_base)
                nyquist_ell_x = predict_nyquist_ell_x(declination)
                ellmax = 0.85 * nyquist_ell_x if np.isfinite(nyquist_ell_x) else np.inf
                source_kmax[source_base] = None if not np.isfinite(ellmax) else float(ellmax / 360.0)  # Keep as k for compatibility

                taper_width_ell = 360.0 * taper_width_cpd
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
                raise ValueError("Fewer than 2 sources available after processing. Cannot build a robust template.")

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
    config.use_precomputed_leakage_templates = False

    # example modifications to the config:
    # config.bands = ["90GHz"]
    # config.coadd_filenames = {"winter": ["/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_tau_decon_winter.g3"]}
    # config.use_cdrc = True
    # config.leakage_template_dir = "/home/tijmen/cmb_analysis/beam_analysis/leakage_templates"

    construct_templates(config, make_plots=True)
