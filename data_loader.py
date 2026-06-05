"""Data loading and preprocessing for the polarized beam fitter."""

import os
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional, Tuple

import numpy as np

from .precision import calculate_precision
from .source_fitting import gaussfit_source
from .utils import (
    check_zero_fraction,
    compute_rectangular_ell_cut_indices,
    make_apodization_mask,
    match_subfield_by_declination,
    shift_map_bilinear,
)


@dataclass
class WeightMaps:
    """Plain NumPy representation of the Stokes weight maps."""

    TT: np.ndarray
    QQ: np.ndarray
    UU: np.ndarray
    TQ: np.ndarray
    TU: np.ndarray
    QU: np.ndarray


@dataclass
class SourceMapRecord:
    """A source cutout converted from G3 containers into plain arrays."""

    source_id: str
    band: str
    t: np.ndarray
    q: np.ndarray
    u: np.ndarray
    weight: WeightMaps
    res: float


def _require_spt3g():
    """Import SPT3G lazily so package import and non-G3 workflows stay usable without it."""
    try:
        from spt3g import core, maps
    except ImportError as err:
        raise ImportError(
            "Reading G3 coadd files requires spt3g_software. "
            "Install it only for workflows that load .g3 files; the rest of polarized_beam_fitting uses NumPy arrays."
        ) from err
    return core, maps


def _band_from_source_id(source_id: str, bands: Iterable[str]) -> Optional[str]:
    """Extract the configured band label from a source ID."""
    for band in bands:
        if band in source_id:
            return band
    return None


def _base_source_id(source_id: str, band: str) -> str:
    """Strip the band suffix from a source ID if present."""
    suffix = f"-{band}"
    return source_id.split(suffix)[0] if suffix in source_id else source_id


def _iter_g3_map_frames(filename: str) -> Iterator:
    """Yield map frames from a G3 file."""
    core, _ = _require_spt3g()
    g3_file = core.G3File(filename)
    for frame in g3_file:
        if frame.type == core.G3FrameType.Map and "Id" in frame:
            yield frame


def iter_g3_source_ids(filename: str, bands: Iterable[str]) -> Iterator[str]:
    """Yield source IDs from a G3 file without materializing map arrays."""
    for frame in _iter_g3_map_frames(filename):
        source_id = str(frame["Id"])
        if _band_from_source_id(source_id, bands) is not None:
            yield source_id


def _extract_source_map_record(frame, band: str, max_zero_fraction: Optional[float] = None) -> Optional[SourceMapRecord]:
    """Convert one G3 source frame to plain NumPy arrays after removing weights."""
    _, maps = _require_spt3g()
    source_id = str(frame["Id"])
    t_map, q_map, u_map, weight = (frame["T"], frame["Q"], frame["U"], frame["Wpol"])
    map_res = float(t_map.res)
    maps.remove_weights(t_map, q_map, u_map, weight, zero_nans=False)

    if max_zero_fraction is not None and not check_zero_fraction(t_map, source_id, max_zero_fraction=max_zero_fraction):
        return None

    weight_maps = WeightMaps(
        TT=np.array(weight.TT, copy=True),
        QQ=np.array(weight.QQ, copy=True),
        UU=np.array(weight.UU, copy=True),
        TQ=np.array(weight.TQ, copy=True),
        TU=np.array(weight.TU, copy=True),
        QU=np.array(weight.QU, copy=True),
    )
    return SourceMapRecord(
        source_id=source_id,
        band=band,
        t=np.array(t_map, copy=True),
        q=np.array(q_map, copy=True),
        u=np.array(u_map, copy=True),
        weight=weight_maps,
        res=map_res,
    )


def load_g3_source_map_records(
    filename: str,
    bands: Iterable[str],
    source_bases: Optional[Iterable[str]] = None,
    max_zero_fraction: Optional[float] = None,
) -> Iterator[SourceMapRecord]:
    """Yield plain source-map records from a G3 file for the requested bands."""
    source_base_set = None if source_bases is None else set(source_bases)
    for frame in _iter_g3_map_frames(filename):
        source_id = str(frame["Id"])
        band = _band_from_source_id(source_id, bands)
        if band is None:
            continue
        if source_base_set is not None and _base_source_id(source_id, band) not in source_base_set:
            continue
        record = _extract_source_map_record(frame, band, max_zero_fraction=max_zero_fraction)
        if record is not None:
            yield record


class DataLoader:
    """Handles data loading and preprocessing."""

    def __init__(self, config):
        self.config = config
        self.map_shape = (config.map_size_pix, config.map_size_pix)

    def load_and_prepare(self) -> Tuple:
        """Main entry point for data loading."""
        self._validate_skip_sources()

        # Load raw data and perform gaussian fits
        if self.config.use_cdrc and self.config.chi2_method != "fourier":
            raise ValueError("CDRC is only supported with chi2_method='fourier'.")
        if self.config.use_cdrc:
            fields = set(self.config.coadd_filenames.keys())
            configured_fields = set(self.config.cdrc_params)
            if not fields.issubset(configured_fields):
                raise ValueError(
                    f"CDRC is not configured for coadd field(s) {sorted(fields - configured_fields)}; "
                    f"configured fields are {sorted(configured_fields)}."
                )
        gaussfit_results = self._load_and_fit_sources()

        # Extract consistent sources across bands
        source_data = self._extract_consistent_sources(gaussfit_results)
        source_fields = source_data["fields"]

        if self.config.use_cdrc:
            self._apply_cdrc_to_source_data(source_data)

        # Create leakage template
        qu_templates, template_offsets, template_flux = self._create_leakage_template(
            source_data["gaussfit_amp"], source_data["raw_maps"], source_fields
        )

        template_flux_array = None

        if template_offsets is not None:
            self._apply_template_offsets(source_data, template_offsets)
            if template_flux is None:
                raise ValueError(
                    "Precomputed leakage templates are missing stored flux parameters. Re-run template construction to embed source fluxes."
                )
            template_flux_array = self._build_template_flux_array(source_data, template_flux)
            source_data["gaussfit_amp"] = template_flux_array.copy()

        # Clean maps
        maps_clean, maps_fft = self._prepare_clean_maps(
            source_data["gaussfit_amp"],
            source_data["raw_maps"],
            qu_templates,
            source_fields,
            source_data["source_ids"],
            source_data["gaussfit_yoff"],
            source_data["gaussfit_xoff"],
            template_flux_array,
        )

        if self.config.chi2_method == "fourier":
            precision, debug = self._build_precision(maps_clean, source_fields)
        else:
            precision = None
            debug = None

        return (
            source_data["gaussfit_yoff"],
            source_data["gaussfit_xoff"],
            source_data["gaussfit_amp"],
            source_data["raw_maps"],
            qu_templates,
            maps_clean,
            source_data["weights"],
            maps_fft,
            source_data["source_ids"],
            source_data["fields"],
            source_data["n_src"],
            precision,
            debug,
        )

    def _build_precision(self, maps_clean: np.ndarray, source_fields: np.ndarray):
        """Compute and package noise-model artifacts for caching."""
        n_src, ny_full, nx_full, n_bands, n_stokes = maps_clean.shape

        idx_y, idx_x = compute_rectangular_ell_cut_indices((ny_full, nx_full), self.config.reso_arcmin, self.config.ellmax)
        ny, nx = len(idx_y), len(idx_x)
        precision = np.zeros((n_src, ny, nx, n_bands, n_stokes, n_bands, n_stokes), dtype=self.config.dtype_np_real)

        # Group sources by field and calculate precision per field
        unique_fields = np.unique(source_fields)
        all_debug = {}

        for field in unique_fields:
            field_mask = source_fields == field
            field_indices = np.where(field_mask)[0]

            assert len(field_indices) > 0, f"No sources found for field '{field}'."

            field_maps_clean = maps_clean[field_indices]
            field_precision, field_debug = calculate_precision(field_maps_clean, self.config)
            precision[field_indices] = field_precision

            if self.config.debug:
                all_debug[field] = field_debug

        return precision, all_debug

    def _normalize_source_id(self, source_id: str) -> str:
        """Extract coordinate portion of source ID, stripping prefixes and band suffixes.

        Examples:
            'J142756-4206.3' -> '142756-4206.3'
            'CoaddSPT-S J142756-4206.3' -> '142756-4206.3'
            'J142756-4206.3-90GHz' -> '142756-4206.3'
            '142756-4206.3-90GHz' -> '142756-4206.3'
        """
        s = source_id
        if "J" in s:
            s = s.split("J", 1)[1]
        for band in self.config.bands:
            s = s.replace(f"-{band}", "")
        return s.strip()

    def _iter_source_ids(self, filename: str) -> Iterator[str]:
        """Yield source IDs from one configured input file."""
        return iter_g3_source_ids(filename, self.config.bands)

    def _iter_source_map_records(self, filename: str) -> Iterator[SourceMapRecord]:
        """Yield source map records from one configured input file."""
        return load_g3_source_map_records(filename, self.config.bands, max_zero_fraction=self.config.max_zero_fraction)

    def _validate_skip_sources(self):
        """Check that skip_sources exist in data files."""
        all_source_ids = set()

        for _, filename in self.config.iter_coadd_files():
            all_source_ids.update(self._iter_source_ids(filename))

        normalized_data_ids = {self._normalize_source_id(sid) for sid in all_source_ids}
        missing = [s for s in self.config.skip_sources if self._normalize_source_id(s) not in normalized_data_ids]

        if missing:
            print(f"WARNING: Skip sources not found: {missing}")

    def _load_and_fit_sources(self) -> Dict:
        """Load data and perform Gaussian fits."""
        results = {band: {} for band in self.config.bands}

        for field_name, filename in self.config.iter_coadd_files():
            print(f"Processing: {filename} (field={field_name})")
            self._process_file(field_name, filename, results)

        return results

    def _process_file(self, field_name: str, filename: str, results: Dict):
        """Process a single coadd file."""
        for source_maps in self._iter_source_map_records(filename):
            source_id = source_maps.source_id
            band = source_maps.band

            if band is None or self._should_skip_source(source_id):
                continue

            if self.config.use_precomputed_leakage_templates:
                fit_result = self._build_precomputed_source_record(source_maps, band, field_name)
            else:
                fit_result = self._fit_single_source(source_maps, band, field_name)
            if fit_result is not None:
                fit_result["field"] = field_name
                results[band][source_id] = fit_result

    def _get_band_from_id(self, source_id: str) -> Optional[str]:
        """Extract band from source ID."""
        return _band_from_source_id(source_id, self.config.bands)

    def _should_skip_source(self, source_id: str) -> bool:
        """Check if source should be skipped."""
        normalized_id = self._normalize_source_id(source_id)
        for skip_source in self.config.skip_sources:
            if self._normalize_source_id(skip_source) == normalized_id:
                print(f"Skipping source {source_id} because it is in the skip_sources list.")
                return True
        return False

    def _get_cdrc_params(self, source_id: str, band: str, field_name: str) -> Tuple[Dict[str, float], str]:
        """Return CDRC parameters and matched subfield name for a source."""
        field_params = self.config.cdrc_params.get(field_name)
        if field_params is None:
            raise ValueError(f"CDRC is not configured for field='{field_name}'.")

        subfield_name = match_subfield_by_declination(source_id, field_params.keys())
        band_params = field_params[subfield_name].get(band)
        if band_params is None:
            raise ValueError(f"No CDRC parameters found for band '{band}' in subfield '{subfield_name}'.")

        return band_params, subfield_name

    def _build_cdrc_transform(self, band: str, params: Dict[str, float]) -> np.ndarray:
        """Build the 3x3 CDRC transform matrix for (T, Q, U)."""
        tcal = float(params.get("tcal", self.config.cmb_calibration_factors["T"][band]))
        pcal = float(params.get("pcal", self.config.cmb_calibration_factors["Q"][band] / self.config.cmb_calibration_factors["T"][band]))
        eps_q = float(params["epsilon_q_tt"])
        eps_u = float(params["epsilon_u_tt"])
        psi = 2.0 * float(params["delta_psi"])

        c = np.cos(psi)
        s = np.sin(psi)

        a1 = np.diag([tcal, tcal, tcal])
        a2 = np.array([[1.0, 0.0, 0.0], [-eps_q, 1.0, 0.0], [-eps_u, 0.0, 1.0]], dtype=self.config.dtype_np_real)
        a3 = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=self.config.dtype_np_real)
        a4 = np.diag([1.0, pcal, pcal])
        return a4 @ a3 @ a2 @ a1

    def _apply_cdrc_to_source_data(self, source_data: Dict) -> None:
        """Apply CDRC transforms to raw maps and per-band flux amplitudes in-place."""
        source_ids = source_data["source_ids"]
        source_fields = source_data["fields"]
        raw_maps = source_data["raw_maps"]
        gaussfit_amp = source_data["gaussfit_amp"]

        n_src, _, _, n_bands, n_stokes = raw_maps.shape
        if n_stokes != 3:
            raise ValueError(f"Expected 3 stokes parameters for CDRC, got {n_stokes}.")

        for i in range(n_src):
            field = source_fields[i]
            source_id = str(source_ids[i])
            for j, band in enumerate(self.config.bands):
                params, subfield_name = self._get_cdrc_params(source_id, band, field)
                transform = self._build_cdrc_transform(band, params).astype(self.config.dtype_np_real)
                raw_maps[i, :, :, j, :] = np.einsum("wv,yxv->yxw", transform, raw_maps[i, :, :, j, :], optimize=True)
                gaussfit_amp[i, j, :] = transform @ gaussfit_amp[i, j, :]
                if self.config.debug:
                    print(f"Applied map-domain CDRC to {source_id} (field={field}, subfield={subfield_name}, band={band}).")

    def _fit_single_source(self, source_maps: SourceMapRecord, band: str, field_name: str) -> Optional[Dict]:
        """Fit Gaussian to a single source."""
        fit = gaussfit_source(
            source_maps.t,
            source_maps.q,
            source_maps.u,
            source_maps.weight,
            config=self.config,
            band=band,
            map_res=source_maps.res,
        )
        yoff, xoff, t_amp, meanoff, q_amp, u_amp = fit

        if t_amp < self.config.min_t_amplitude:
            print(f"Skipping source {source_maps.source_id} because T amplitude is {t_amp}<{self.config.min_t_amplitude}.")
            return None

        return {
            "yoff": yoff,
            "xoff": xoff,
            "t_amp": t_amp,
            "q_amp": q_amp,
            "u_amp": u_amp,
            "meanoff": meanoff,
            "maps": (source_maps.t, source_maps.q, source_maps.u, source_maps.weight),
        }

    def _build_precomputed_source_record(self, source_maps: SourceMapRecord, band: str, field_name: str) -> Optional[Dict]:
        """Collect maps without running Gaussian fits (for precomputed templates)."""
        zero = float(0.0)
        return {
            "yoff": zero,
            "xoff": zero,
            "t_amp": zero,
            "q_amp": zero,
            "u_amp": zero,
            "meanoff": zero,
            "maps": (source_maps.t, source_maps.q, source_maps.u, source_maps.weight),
        }

    def _extract_consistent_sources(self, gaussfit_results: Dict) -> Dict:
        """Extract sources that have data for all bands."""
        source_ids = self._find_common_sources(gaussfit_results)
        n_src = len(source_ids)
        n_bands = len(self.config.bands)
        ny, nx = self.map_shape

        arrays = {
            "gaussfit_yoff": np.zeros((n_src, n_bands), dtype=self.config.dtype_np_real),
            "gaussfit_xoff": np.zeros((n_src, n_bands), dtype=self.config.dtype_np_real),
            "gaussfit_amp": np.zeros((n_src, n_bands, 3), dtype=self.config.dtype_np_real),
            "raw_maps": np.zeros((n_src, ny, nx, n_bands, 3), dtype=self.config.dtype_np_real),
            "weights": np.zeros((n_src, ny, nx, n_bands, 3, 3), dtype=self.config.dtype_np_real),
            "source_ids": source_ids,
            "n_src": n_src,
            "fields": np.empty(n_src, dtype=object),
        }

        self._fill_source_arrays(arrays, gaussfit_results, source_ids)
        return arrays

    def _find_common_sources(self, gaussfit_results: Dict) -> np.ndarray:
        """Find sources present in all bands."""
        base_band = self.config.bands[0]
        source_ids = list(gaussfit_results[base_band].keys())

        if len(self.config.bands) > 1:
            source_ids = [sid for sid in source_ids if self._source_in_all_bands(sid, gaussfit_results)]

        base_names = [sid.split(f"-{base_band}")[0] for sid in source_ids]
        return np.array(sorted(base_names))

    def _source_in_all_bands(self, source_id: str, results: Dict) -> bool:
        """Check if source exists in all bands."""
        base_name = source_id.split(f"-{self.config.bands[0]}")[0]

        for band in self.config.bands[1:]:
            expected_id = f"{base_name}-{band}"
            if expected_id not in results[band]:
                return False
        return True

    def _fill_source_arrays(self, arrays: Dict, results: Dict, source_ids: np.ndarray):
        """Fill numpy arrays with source data."""
        for i, sid in enumerate(source_ids):
            for j, band in enumerate(self.config.bands):
                result = results[band][f"{sid}-{band}"]
                self._fill_band_data(arrays, i, j, result)
                if j == 0:
                    field_value = result.get("field")
                    if field_value is None:
                        raise ValueError(f"Missing field metadata for source '{sid}'.")
                    arrays["fields"][i] = field_value

    def _fill_band_data(self, arrays: Dict, src_idx: int, band_idx: int, result: Dict):
        """Fill arrays for a single band of a source."""
        arrays["gaussfit_yoff"][src_idx, band_idx] = result["yoff"]
        arrays["gaussfit_xoff"][src_idx, band_idx] = result["xoff"]
        arrays["gaussfit_amp"][src_idx, band_idx, 0] = result["t_amp"]
        arrays["gaussfit_amp"][src_idx, band_idx, 1] = result["q_amp"]
        arrays["gaussfit_amp"][src_idx, band_idx, 2] = result["u_amp"]

        arrays["raw_maps"][src_idx, :, :, band_idx, 0] = result["maps"][0]
        arrays["raw_maps"][src_idx, :, :, band_idx, 1] = result["maps"][1]
        arrays["raw_maps"][src_idx, :, :, band_idx, 2] = result["maps"][2]

        w = result["maps"][3]
        arrays["weights"][src_idx, :, :, band_idx, 0, 0] = w.TT
        arrays["weights"][src_idx, :, :, band_idx, 1, 1] = w.QQ
        arrays["weights"][src_idx, :, :, band_idx, 2, 2] = w.UU
        arrays["weights"][src_idx, :, :, band_idx, 0, 1] = w.TQ
        arrays["weights"][src_idx, :, :, band_idx, 1, 0] = w.TQ
        arrays["weights"][src_idx, :, :, band_idx, 0, 2] = w.TU
        arrays["weights"][src_idx, :, :, band_idx, 2, 0] = w.TU
        arrays["weights"][src_idx, :, :, band_idx, 1, 2] = w.QU
        arrays["weights"][src_idx, :, :, band_idx, 2, 1] = w.QU

    def _create_leakage_template(self, gaussfit_amp: np.ndarray, raw_maps: np.ndarray, source_fields: np.ndarray) -> Tuple:
        """Create or load the T->QU leakage correction template(s) with offsets and fluxes."""
        if self.config.use_precomputed_leakage_templates:
            return self._load_precomputed_leakage_templates(source_fields)

        print(f"Creating T->P leakage template ({self.config.leakage_weighting} weighting)...")

        qu_maps = raw_maps[:, :, :, :, 1:3]
        t_amps = gaussfit_amp[:, :, 0]

        if self.config.leakage_weighting == "median":
            template = np.median(qu_maps / t_amps[:, None, None, :, None], axis=0)
        else:
            weight_map = {"flat": 1.0, "linear": t_amps, "quadratic": t_amps**2}
            weights_base = weight_map[self.config.leakage_weighting]

            if self.config.leakage_weighting == "flat":
                weights = np.ones_like(t_amps)[:, None, None, :, None]
            else:
                weights = weights_base[:, None, None, :, None]

            normalized_qu = qu_maps / t_amps[:, None, None, :, None]
            weighted_sum = np.sum(normalized_qu * weights, axis=0)
            weight_sum = np.sum(weights, axis=0)
            template = weighted_sum / weight_sum

        return template, None, None

    def _load_precomputed_leakage_templates(
        self, source_fields: np.ndarray
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
        Dict[str, Dict[str, Dict[str, np.ndarray]]],
    ]:
        """Load precomputed leakage templates, per-band source offsets, and stored fluxes."""
        unique_fields = {field for field in source_fields if field is not None}
        if not unique_fields:
            raise ValueError("No source fields available to match precomputed leakage templates.")

        templates = {}
        template_offsets: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}
        template_fluxes: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        ny, nx = self.map_shape
        n_bands = len(self.config.bands)

        for field in unique_fields:
            field_templates = np.zeros((ny, nx, n_bands, 2), dtype=self.config.dtype_np_real)
            field_offsets: Dict[str, Dict[str, Tuple[float, float]]] = template_offsets.setdefault(field, {})
            field_fluxes: Dict[str, Dict[str, np.ndarray]] = template_fluxes.setdefault(field, {})

            for band_idx, band in enumerate(self.config.bands):
                suffix = "_cdrc" if self.config.use_cdrc else ""
                filename = os.path.join(
                    self.config.leakage_template_dir,
                    f"leakage_template_{field}_{band}_{self.config.leakage_weighting}{suffix}.pkl",
                )

                if not os.path.exists(filename):
                    raise FileNotFoundError(
                        f"Missing leakage template for field '{field}', band '{band}', weighting '{self.config.leakage_weighting}': {filename}"
                    )

                with open(filename, "rb") as f:
                    template_data = pickle.load(f)

                q_map = np.asarray(template_data["Q"], dtype=self.config.dtype_np_real)
                u_map = np.asarray(template_data["U"], dtype=self.config.dtype_np_real)

                if q_map.shape != (ny, nx) or u_map.shape != (ny, nx):
                    raise ValueError(f"Template shape mismatch in {filename}: expected {(ny, nx)}, got {q_map.shape}/{u_map.shape}")

                field_templates[:, :, band_idx, 0] = q_map
                field_templates[:, :, band_idx, 1] = u_map

                offsets_payload = template_data.get("source_offsets")
                if offsets_payload is None:
                    raise ValueError(f"Template file {filename} does not contain 'source_offsets'.")
                flux_payload = template_data.get("source_flux")
                if flux_payload is None:
                    raise ValueError(
                        f"Template file {filename} does not contain 'source_flux'. Re-run template construction to include stored flux parameters."
                    )

                for source_id, offsets in offsets_payload.items():
                    y_offset = float(offsets["y_offset"])
                    x_offset = float(offsets["x_offset"])

                    # Always trust the offsets stored with each template; they supersede gaussfit positions.
                    src = str(source_id)
                    suffix = f"-{band}"
                    base_id = src.split(suffix)[0] if suffix in src else src
                    per_source_offsets = field_offsets.setdefault(base_id, {})
                    per_source_offsets[band] = (y_offset, x_offset)

                for source_id, band_flux_map in flux_payload.items():
                    src = str(source_id)
                    suffix = f"-{band}"
                    base_id = src.split(suffix)[0] if suffix in src else src
                    per_source_flux = field_fluxes.setdefault(base_id, {})
                    flux_entry = band_flux_map.get(band)
                    if flux_entry is None:
                        raise ValueError(f"Template file {filename} does not contain stored flux for source '{base_id}' and band '{band}'.")
                    flux_vector = np.array(
                        [float(flux_entry["T"]), float(flux_entry["Q"]), float(flux_entry["U"])],
                        dtype=self.config.dtype_np_real,
                    )
                    per_source_flux[band] = flux_vector

            templates[field] = field_templates

        return templates, template_offsets, template_fluxes

    def _apply_template_offsets(self, source_data: Dict, template_offsets: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]):
        """Overwrite initial offsets with the values stored alongside precomputed templates."""
        gaussfit_yoff = source_data["gaussfit_yoff"]  # reference to mutable array
        gaussfit_xoff = source_data["gaussfit_xoff"]
        source_ids = source_data["source_ids"]
        source_fields = source_data["fields"]

        for idx, raw_source_id in enumerate(source_ids):
            sid = str(raw_source_id)
            field = source_fields[idx]
            field_offsets = template_offsets.get(field)
            if field_offsets is None:
                raise ValueError(f"Missing stored offsets for field '{field}' while using precomputed templates.")
            source_offset_map = field_offsets.get(sid)
            if source_offset_map is None:
                raise ValueError(f"Missing stored offset for source '{sid}' in field '{field}' while using precomputed templates.")
            for band_idx, band in enumerate(self.config.bands):
                if band not in source_offset_map:
                    raise ValueError(f"Missing stored offset for source '{sid}' in field '{field}' for band '{band}'.")
                y_offset, x_offset = source_offset_map[band]
                gaussfit_yoff[idx, band_idx] = y_offset
                gaussfit_xoff[idx, band_idx] = x_offset

    def _build_template_flux_array(self, source_data: Dict, template_flux: Dict[str, Dict[str, Dict[str, np.ndarray]]]) -> np.ndarray:
        """Assemble stored flux parameters into an array aligned with the source ordering."""
        flux_array = np.zeros(source_data["gaussfit_amp"].shape, dtype=self.config.dtype_np_real)
        source_ids = source_data["source_ids"]
        source_fields = source_data["fields"]

        for idx, raw_source_id in enumerate(source_ids):
            sid = str(raw_source_id)
            field = source_fields[idx]
            field_flux = template_flux.get(field)
            if field_flux is None:
                raise ValueError(f"Missing stored flux for field '{field}' while using precomputed templates.")
            source_flux_map = field_flux.get(sid)
            if source_flux_map is None:
                raise ValueError(f"Missing stored flux for source '{sid}' in field '{field}'.")

            for band_idx, band in enumerate(self.config.bands):
                flux_vector = source_flux_map.get(band)
                if flux_vector is None:
                    raise ValueError(
                        f"Missing stored flux for source '{sid}' in field '{field}' for band '{band}'. "
                        "Re-run template construction to regenerate templates."
                    )
                flux_array[idx, band_idx, :] = flux_vector

        return flux_array

    def _prepare_clean_maps(
        self,
        gaussfit_amp: np.ndarray,
        raw_maps: np.ndarray,
        qu_templates,
        source_fields: np.ndarray,
        source_ids: np.ndarray,
        gaussfit_yoff: np.ndarray,
        gaussfit_xoff: np.ndarray,
        template_flux: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply leakage correction, subtract shifted templates, and prepare final maps."""
        n_src, ny, nx, n_bands, _ = raw_maps.shape
        maps_clean = raw_maps.copy()

        for i in range(n_src):
            if self.config.use_precomputed_leakage_templates:
                field = source_fields[i]
                if field not in qu_templates:
                    raise ValueError(f"No precomputed leakage template available for field '{field}'.")
                field_template = qu_templates[field]
            else:
                field_template = qu_templates

            for j in range(n_bands):
                template_q = field_template[:, :, j, 0]
                template_u = field_template[:, :, j, 1]

                if self.config.use_precomputed_leakage_templates:
                    y_offset = float(gaussfit_yoff[i, j])
                    x_offset = float(gaussfit_xoff[i, j])
                    if abs(y_offset) > 1e-6 or abs(x_offset) > 1e-6:
                        template_q_shifted = shift_map_bilinear(template_q, y_offset, x_offset)
                        template_u_shifted = shift_map_bilinear(template_u, y_offset, x_offset)
                    else:
                        template_q_shifted = template_q
                        template_u_shifted = template_u
                else:
                    template_q_shifted = template_q
                    template_u_shifted = template_u

                if self.config.use_precomputed_leakage_templates:
                    if template_flux is None:
                        raise ValueError(
                            "Precomputed leakage templates require stored flux parameters for scaling. "
                            "Re-run template construction to embed flux information."
                        )
                    t_amp = float(template_flux[i, j, 0])
                else:
                    t_amp = gaussfit_amp[i, j, 0]

                maps_clean[i, :, :, j, 1] -= template_q_shifted * t_amp
                maps_clean[i, :, :, j, 2] -= template_u_shifted * t_amp

        maps_fft = None
        if self.config.chi2_method == "fourier":
            apod_mask = make_apodization_mask(self.map_shape, self.config.apodization_width_pix)
            apod_mask = apod_mask.astype(self.config.dtype_np_real)
            apodized = maps_clean * apod_mask[None, :, :, None, None]
            maps_fft = np.fft.fft2(apodized, axes=(1, 2))
            idx_y, idx_x = compute_rectangular_ell_cut_indices(self.map_shape, self.config.reso_arcmin, self.config.ellmax)
            maps_fft = maps_fft[:, idx_y, :, :, :][:, :, idx_x, :, :]

        print(f"Prepared {n_src} sources for fitting.")
        return maps_clean, maps_fft


class ExampleExperimentDataLoader(DataLoader):
    """
    Example adapter for a non-SPT experiment.

    This class is not used by the SPT workflow. It is a concrete template for
    users who want to feed another experiment's cutout maps into the same
    fitter. Copy it, rename it, and replace the NPZ parsing with the relevant
    experiment-specific file reader.

    The example NPZ schema is:

    - ``source_id``: ``(n_record,)`` strings. IDs should be stable across bands.
      If the ID does not already end with ``-{band}``, the loader appends that
      suffix because the fitter matches multiband sources by that convention.
    - ``band``: optional ``(n_record,)`` strings such as ``150GHz``. If omitted,
      the band is inferred from ``source_id``.
    - ``T``, ``Q``, ``U``: ``(n_record, ny, nx)`` maps in mK.
    - ``WTT``, ``WQQ``, ``WUU``: ``(n_record, ny, nx)`` inverse-variance weights
      in 1 / mK^2.
    - ``WTQ``, ``WTU``, ``WQU``: optional off-diagonal weights, also in 1 / mK^2.
      Missing off-diagonal weights are treated as zero.
    - ``res_rad``: scalar or ``(n_record,)`` pixel resolution in radians.
    """

    def _iter_source_ids(self, filename: str) -> Iterator[str]:
        """Yield source IDs without loading full maps."""
        with np.load(filename, allow_pickle=False) as data:
            source_ids = np.asarray(data["source_id"]).astype(str)
            bands = self._bands_from_npz(data, len(source_ids))
            for source_id, band in zip(source_ids, bands):
                if band is not None:
                    yield self._ensure_band_suffix(source_id, band)

    def _iter_source_map_records(self, filename: str) -> Iterator[SourceMapRecord]:
        """Yield ``SourceMapRecord`` objects from the example NPZ schema."""
        with np.load(filename, allow_pickle=False) as data:
            source_ids = np.asarray(data["source_id"]).astype(str)
            bands = self._bands_from_npz(data, len(source_ids))

            for idx, raw_source_id in enumerate(source_ids):
                band = bands[idx]
                if band is None:
                    continue

                source_id = self._ensure_band_suffix(raw_source_id, band)
                t_map = np.asarray(data["T"][idx], dtype=self.config.dtype_np_real)
                if not check_zero_fraction(t_map, source_id, max_zero_fraction=self.config.max_zero_fraction):
                    continue

                yield SourceMapRecord(
                    source_id=source_id,
                    band=band,
                    t=t_map,
                    q=np.asarray(data["Q"][idx], dtype=self.config.dtype_np_real),
                    u=np.asarray(data["U"][idx], dtype=self.config.dtype_np_real),
                    weight=WeightMaps(
                        TT=np.asarray(data["WTT"][idx], dtype=self.config.dtype_np_real),
                        QQ=np.asarray(data["WQQ"][idx], dtype=self.config.dtype_np_real),
                        UU=np.asarray(data["WUU"][idx], dtype=self.config.dtype_np_real),
                        TQ=self._optional_weight(data, "WTQ", idx, t_map.shape),
                        TU=self._optional_weight(data, "WTU", idx, t_map.shape),
                        QU=self._optional_weight(data, "WQU", idx, t_map.shape),
                    ),
                    res=self._resolution_from_npz(data, idx),
                )

    def _bands_from_npz(self, data, n_record: int) -> list[Optional[str]]:
        """Return band labels, falling back to source ID parsing when needed."""
        if "band" in data.files:
            raw_bands = np.asarray(data["band"]).astype(str)
            return [band if band in self.config.bands else None for band in raw_bands]
        source_ids = np.asarray(data["source_id"]).astype(str)
        return [self._get_band_from_id(source_id) for source_id in source_ids[:n_record]]

    def _ensure_band_suffix(self, source_id: str, band: str) -> str:
        """Return an ID compatible with the fitter's multiband matching convention."""
        suffix = f"-{band}"
        return source_id if source_id.endswith(suffix) else f"{source_id}{suffix}"

    def _optional_weight(self, data, key: str, idx: int, shape: Tuple[int, int]) -> np.ndarray:
        """Read an optional off-diagonal weight map, defaulting to zeros."""
        if key in data.files:
            return np.asarray(data[key][idx], dtype=self.config.dtype_np_real)
        return np.zeros(shape, dtype=self.config.dtype_np_real)

    def _resolution_from_npz(self, data, idx: int) -> float:
        """Read scalar or per-record pixel resolution in radians."""
        res_rad = np.asarray(data["res_rad"], dtype=float)
        if res_rad.ndim == 0:
            return float(res_rad)
        return float(res_rad[idx])
