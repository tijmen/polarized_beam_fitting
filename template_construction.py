"""Construct shifted T->P leakage templates for multiple fields and bands.

This script reproduces the template construction workflow that previously
lived in the notebook, but makes it repeatable via a command-line entry
point. It applies a radial low-pass filter using the TOD Nyquist estimate,
recenters residual maps with bilinear interpolation, and stores source
offsets alongside each template so they can be re-used later.

Run with:
```bash
python -m polarized_beam_fitting.template_construction
```
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from spt3g import core, maps

from polarized_beam_fitting import PolarizedBeamFitter
from polarized_beam_fitting.beam_model import create_beam_model
from polarized_beam_fitting.config import BeamFittingConfig
from polarized_beam_fitting.utils import (
    apply_radial_lowpass,
    make_apodization_mask,
    parse_declination,
    predict_nyquist_kx,
    shift_map_bilinear,
)

# Default bands and coadd files to process
BANDS: Tuple[str, ...] = ("90GHz", "150GHz", "220GHz")
FIELD_COADD_PATHS = {
    "winter": "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_tau_decon_winter.g3",
    "summer_a": "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_tau_decon_summera.g3",
    "summer_b": "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_tau_decon_summerb.g3",
    "summer_c": "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_tau_decon_summerc.g3",
    "winter_nodecon": "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_19-20_winter.g3",
    "summer_a_nodecon": "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_summera.g3",
    "summer_b_nodecon": "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_summerb.g3",
    "summer_c_nodecon": "/home/tijmen/cmb_analysis/beam_analysis/data/bright_thumb_coadd_subfieldall_masked_thumbnails_res0p1_summerc.g3",
}
OUTPUT_DIR = Path("output/leakage_templates")
WEIGHTING_SCHEMES = ("median", "flat", "linear", "quadratic")


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
    """Pre-compute |k| grid and taper width for filtering."""
    ny, nx = map_shape
    reso_deg = reso_arcmin / 60.0
    ky_freq = np.fft.fftfreq(ny, d=reso_deg)
    kx_freq = np.fft.fftfreq(nx, d=reso_deg)
    ky_grid, kx_grid = np.meshgrid(ky_freq, kx_freq, indexing="ij")
    k_radius = np.sqrt(kx_grid**2 + ky_grid**2)

    freq_steps: Iterable[float] = []
    if ny > 1:
        freq_steps = [abs(ky_freq[1] - ky_freq[0])]
    if nx > 1:
        freq_steps = list(freq_steps) + [abs(kx_freq[1] - kx_freq[0])]

    taper_width_cpd = 0.0
    if freq_steps:
        taper_width_cpd = 5.0 * min(freq_steps)
    return k_radius, reso_deg, taper_width_cpd


def load_raw_maps_for_band(config: BeamFittingConfig, fitter: PolarizedBeamFitter) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract raw T/Q/U maps from the G3 file for sources present in the fitter state."""
    raw_maps_data: Dict[str, Dict[str, np.ndarray]] = {}
    band = config.bands[0]
    source_ids = set(fitter.state.source_ids)

    filename = config.coadd_filenames[0]
    try:
        g3_file = core.G3File(filename)
    except RuntimeError as err:
        raise RuntimeError(f"Could not open coadd file {filename}") from err

    for frame in g3_file:
        if frame.type != core.G3FrameType.Map or "Id" not in frame:
            continue

        if band not in frame["Id"]:
            continue

        source_base = frame["Id"].split(f"-{band}")[0]
        if source_base not in source_ids:
            continue

        t_map, q_map, u_map, weight = frame["T"], frame["Q"], frame["U"], frame["Wpol"]
        maps.remove_weights(t_map, q_map, u_map, weight, zero_nans=False)
        raw_maps_data[source_base] = {
            "T": np.array(t_map, copy=False),
            "Q": np.array(q_map, copy=False),
            "U": np.array(u_map, copy=False),
        }

    if not raw_maps_data:
        raise ValueError(f"No sources found in {filename} for band {band}.")
    return raw_maps_data


def construct_templates_for_combination(field: str, band: str, output_dir: Path, make_plots: bool) -> None:
    """Build templates for a single (field, band) combination."""
    print(f"Processing field={field}, band={band} ...")

    config = BeamFittingConfig()
    config.bands = [band]
    config.coadd_filenames = [FIELD_COADD_PATHS[field]]
    config.use_precomputed_leakage_templates = True  # for subsequent iterations. Set to False for the first run

    fitter = PolarizedBeamFitter(config)
    best_fit_params = fitter.run_fit()
    all_source_ids = np.asarray(fitter.state.source_ids)

    map_shape = (config.map_size_pix, config.map_size_pix)
    y_coords = np.arange(-map_shape[0] // 2, map_shape[0] // 2)
    x_coords = np.arange(-map_shape[1] // 2, map_shape[1] // 2)
    y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing="ij")
    beam_model = create_beam_model(config, y_grid, x_grid, band)
    band_beam_params = best_fit_params["beams"][0]

    apod_mask = make_apodization_mask(map_shape, config.apodization_width_pix)
    k_radius, _, taper_width_cpd = prepare_frequency_grids(map_shape, config.reso_arcmin)
    raw_maps_data = load_raw_maps_for_band(config, fitter)

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
        nyquist_kx = predict_nyquist_kx(declination)
        k_max_cpd = 0.85 * nyquist_kx if np.isfinite(nyquist_kx) else np.inf
        source_kmax[source_base] = None if not np.isfinite(k_max_cpd) else float(k_max_cpd)

        q_filtered = apply_radial_lowpass(q_residual, apod_mask, k_radius, k_max_cpd, taper_width_cpd)
        u_filtered = apply_radial_lowpass(u_residual, apod_mask, k_radius, k_max_cpd, taper_width_cpd)

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

    plotting_title = f"Filtered Leakage Templates\n{field} - {band}"
    if make_plots:
        plot_path = output_dir / f"templates_{field}_{band}.png"
        plot_templates(templates, plotting_title, plot_path)

    metadata = {
        "coadd": field,
        "band": band,
        "apod_width_pix": config.apodization_width_pix,
        "taper_width_pixels": 5,
        "reso_arcmin": config.reso_arcmin,
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

        output_path = output_dir / f"leakage_template_{field}_{band}_{scheme}.pkl"
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle)
        print(f"  Saved {scheme} template to {output_path}")


def main():
    fields = sorted(FIELD_COADD_PATHS.keys())

    for field in fields:
        if field not in FIELD_COADD_PATHS:
            raise ValueError(f"Unknown field '{field}'. Known fields: {sorted(FIELD_COADD_PATHS.keys())}")
        if not os.path.exists(FIELD_COADD_PATHS[field]):
            raise FileNotFoundError(f"Coadd file for field '{field}' does not exist: {FIELD_COADD_PATHS[field]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for field in fields:
        for band in BANDS:
            construct_templates_for_combination(field, band, OUTPUT_DIR, make_plots=True)


if __name__ == "__main__":
    main()
