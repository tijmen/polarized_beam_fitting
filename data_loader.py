"""Data loading and preprocessing for the polarized beam fitter."""

import os
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
from spt3g import core, maps

from .noise_psd import create_noise_psd_calculator
from .source_fitting import gaussfit_source
from .utils import check_zero_fraction, compute_rectangular_ell_cut_indices, make_apodization_mask, shift_map_bilinear


class DataLoader:
    """Handles data loading and preprocessing."""

    def __init__(self, config):
        self.config = config
        self.map_shape = (config.map_size_pix, config.map_size_pix)

    def load_and_prepare(self) -> Tuple:
        """Main entry point for data loading."""
        self._validate_skip_sources()

        # Load raw data and perform gaussian fits
        gaussfit_results = self._load_and_fit_sources()

        # Extract consistent sources across bands
        source_data = self._extract_consistent_sources(gaussfit_results)
        source_fields = source_data["fields"]

        # Create leakage template
        qu_templates, template_offsets = self._create_leakage_template(source_data["gaussfit_amp"], source_data["raw_maps"], source_fields)

        if template_offsets is not None:
            self._apply_template_offsets(source_data, template_offsets)

        # Clean maps
        maps_clean, maps_fft = self._prepare_clean_maps(
            source_data["gaussfit_amp"],
            source_data["raw_maps"],
            qu_templates,
            source_fields,
            source_data["gaussfit_yoff"],
            source_data["gaussfit_xoff"],
        )

        maps_fft_prepared, noise_bundle = self._build_noise_bundle(maps_clean, maps_fft, source_fields)

        return (
            source_data["gaussfit_yoff"],
            source_data["gaussfit_xoff"],
            source_data["gaussfit_amp"],
            source_data["raw_maps"],
            qu_templates,
            maps_clean,
            source_data["weights"],
            maps_fft_prepared,
            source_data["source_ids"],
            source_data["fields"],
            source_data["n_src"],
            noise_bundle,
        )

    def _build_noise_bundle(self, maps_clean: np.ndarray, maps_fft: Optional[np.ndarray], source_fields: np.ndarray):
        """Compute and package noise-model artifacts for caching."""
        bundle: Dict[str, Any] = {
            "method": self.config.noise_psd_method,
            "precision": None,
            "k_indices_y": None,
            "k_indices_x": None,
            "calculator_payload": None,
        }

        if self.config.chi2_method != "fourier" or maps_fft is None:
            return maps_fft, bundle

        idx_y, idx_x = compute_rectangular_ell_cut_indices(
            (maps_fft.shape[1], maps_fft.shape[2]),
            self.config.reso_arcmin,
            getattr(self.config, "ellmax", None),
        )

        psd_calc = create_noise_psd_calculator(self.config, self.map_shape)
        precision = self._compute_precision_per_source(psd_calc, maps_clean, source_fields, idx_y, idx_x)
        maps_fft_prepared = self._truncate_fourier_numpy(maps_fft, idx_y, idx_x, axis_y=1, axis_x=2)

        bundle["precision"] = precision
        bundle["k_indices_y"] = idx_y
        bundle["k_indices_x"] = idx_x
        bundle["calculator_payload"] = getattr(psd_calc, "payload", None)
        return maps_fft_prepared, bundle

    def _compute_precision_per_source(
        self,
        psd_calc,
        maps_clean: np.ndarray,
        source_fields: np.ndarray,
        idx_y: Optional[np.ndarray],
        idx_x: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Return precision tensors per source for Fourier-space analyses."""
        if self.config.chi2_method != "fourier":
            return None

        method = self.config.noise_psd_method
        if method == "multiband_covariance":
            return self._precision_from_multiband_covariance(psd_calc, maps_clean, source_fields, idx_y, idx_x)
        if method == "pca_multiband_covariance":
            return self._precision_from_pca_multiband(psd_calc, maps_clean, source_fields, idx_y, idx_x)
        if method == "parametric_precision":
            precision = np.asarray(psd_calc.calculate_noise_psd(maps_clean)).astype(self.config.dtype_np_complex)
            precision = self._truncate_fourier_numpy(precision, idx_y, idx_x, axis_y=1, axis_x=2)
            return precision

        psd = np.asarray(psd_calc.calculate_noise_psd(maps_clean)).astype(self.config.dtype_np_real)
        n_src = maps_clean.shape[0]
        if psd.ndim == 4:
            psd = np.broadcast_to(psd, (n_src,) + psd.shape)

        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.reciprocal(psd)
        precision = np.where(np.isfinite(precision), precision, 0.0).astype(self.config.dtype_np_real)
        precision = self._truncate_fourier_numpy(precision, idx_y, idx_x, axis_y=1, axis_x=2)
        return precision

    def _precision_from_multiband_covariance(
        self,
        psd_calc,
        maps_clean: np.ndarray,
        source_fields: np.ndarray,
        idx_y: Optional[np.ndarray],
        idx_x: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compute per-source precision matrices for multiband covariance methods."""
        unique_fields = np.unique(source_fields)
        precision_by_field: Dict[str, np.ndarray] = {}

        for field in unique_fields:
            field_indices = np.where(source_fields == field)[0]
            field_maps = maps_clean[field_indices]
            covariance_psd = np.asarray(psd_calc.calculate_noise_psd(field_maps)).astype(self.config.dtype_np_real)
            precision_grid = self._invert_covariance_stack(covariance_psd)
            precision_grid = self._truncate_fourier_numpy(precision_grid, idx_y, idx_x, axis_y=0, axis_x=1)
            precision_by_field[field] = precision_grid

        sample = next(iter(precision_by_field.values()))
        precision_per_source = np.zeros((maps_clean.shape[0],) + sample.shape, dtype=sample.dtype)

        for field, precision_grid in precision_by_field.items():
            indices = np.where(source_fields == field)[0]
            repeated = np.broadcast_to(precision_grid, (len(indices),) + precision_grid.shape)
            precision_per_source[indices] = repeated

        return precision_per_source

    def _precision_from_pca_multiband(
        self,
        psd_calc,
        maps_clean: np.ndarray,
        source_fields: np.ndarray,
        idx_y: Optional[np.ndarray],
        idx_x: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compile PCA-based precision matrices per source."""
        unique_fields = np.unique(source_fields)
        precision_by_field: Dict[str, np.ndarray] = {}

        for field in unique_fields:
            field_indices = np.where(source_fields == field)[0]
            field_maps = maps_clean[field_indices]
            precision_grid = np.asarray(psd_calc.calculate_noise_psd(field_maps)).astype(self.config.dtype_np_complex)
            precision_grid = self._truncate_fourier_numpy(precision_grid, idx_y, idx_x, axis_y=0, axis_x=1)
            precision_by_field[field] = precision_grid

        sample = next(iter(precision_by_field.values()))
        precision_per_source = np.zeros((maps_clean.shape[0],) + sample.shape, dtype=sample.dtype)

        for field, precision_grid in precision_by_field.items():
            indices = np.where(source_fields == field)[0]
            repeated = np.broadcast_to(precision_grid, (len(indices),) + precision_grid.shape)
            precision_per_source[indices] = repeated

        return precision_per_source

    def _truncate_fourier_numpy(self, array: Optional[np.ndarray], idx_y, idx_x, axis_y: int, axis_x: int):
        """Return array truncated along specified Fourier axes if indices are provided."""
        if array is None or idx_y is None or idx_x is None:
            return array
        truncated = np.take(array, idx_y, axis=axis_y)
        truncated = np.take(truncated, idx_x, axis=axis_x)
        return truncated

    def _project_to_spd(self, matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Project a covariance matrix to the nearest symmetric positive-definite matrix."""
        symm = 0.5 * (matrix + matrix.T)
        diag = np.diag(symm)
        positive_diag = diag[diag > 0]
        scale = np.median(positive_diag) if positive_diag.size else 1.0
        floor = eps * scale
        eigvals, eigvecs = np.linalg.eigh(symm)
        eigvals = np.maximum(eigvals, floor)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _invert_covariance_stack(self, covariance_psd: np.ndarray) -> np.ndarray:
        """Invert a stack of covariance matrices with regularization."""
        ny, nx = covariance_psd.shape[:2]
        n_bands = len(self.config.bands)
        n_stokes = 3
        n_dim = n_bands * n_stokes
        precision = np.zeros((ny, nx, n_bands, n_stokes, n_bands, n_stokes), dtype=self.config.dtype_np_complex)

        for iy in range(ny):
            for ix in range(nx):
                cov = covariance_psd[iy, ix].reshape(n_dim, n_dim)
                cov_spd = self._project_to_spd(cov, eps=1e-5)
                try:
                    prec = np.linalg.inv(cov_spd)
                except np.linalg.LinAlgError:
                    prec = np.linalg.pinv(cov_spd)
                prec = 0.5 * (prec + prec.T)
                precision[iy, ix] = prec.reshape(n_bands, n_stokes, n_bands, n_stokes)

        return precision

    def _validate_skip_sources(self):
        """Check that skip_sources exist in data files."""
        all_source_ids = set()

        for filename in self.config.coadd_filenames:
            try:
                g3_file = core.G3File(filename)
            except RuntimeError:
                print(f"Warning: Could not open {filename}. Skipping.")
                continue

            for frame in g3_file:
                if frame.type == core.G3FrameType.Map and "Id" in frame:
                    if any(band in frame["Id"] for band in self.config.bands):
                        all_source_ids.add(frame["Id"])

        missing = [s for s in self.config.skip_sources if not any(s in sid for sid in all_source_ids)]

        if missing:
            print(f"WARNING: Skip sources not found: {missing}")

    def _load_and_fit_sources(self) -> Dict:
        """Load data and perform Gaussian fits."""
        results = {band: {} for band in self.config.bands}

        for filename in self.config.coadd_filenames:
            print(f"Processing: {filename}")
            self._process_file(filename, results)

        return results

    def _process_file(self, filename: str, results: Dict):
        """Process a single coadd file."""
        try:
            g3_file = core.G3File(filename)
        except RuntimeError:
            print(f"Warning: Could not open {filename}. Skipping.")
            return

        field_name = self._infer_field_from_filename(filename)

        for frame in g3_file:
            if frame.type != core.G3FrameType.Map or "Id" not in frame:
                continue

            source_id = frame["Id"]
            band = self._get_band_from_id(source_id)

            if band is None or self._should_skip_source(source_id):
                continue

            fit_result = self._fit_single_source(frame)
            if fit_result is not None:
                fit_result["field"] = field_name
                results[band][source_id] = fit_result

    def _infer_field_from_filename(self, filename: str) -> str:
        """Infer observing field (winter/summer_a/...) from the input filename."""
        basename = os.path.basename(filename).lower()
        if "winter" in basename:
            return "winter"
        if "summera" in basename:
            return "summer_a"
        if "summerb" in basename:
            return "summer_b"
        if "summerc" in basename:
            return "summer_c"
        else:
            print(f"Warning, could not infer field from filename: {filename}, assuming winter field.")
            return "winter"

    def _get_band_from_id(self, source_id: str) -> Optional[str]:
        """Extract band from source ID."""
        for band in self.config.bands:
            if band in source_id:
                return band
        return None

    def _should_skip_source(self, source_id: str) -> bool:
        """Check if source should be skipped."""
        return any(skip in source_id for skip in self.config.skip_sources)

    def _fit_single_source(self, frame) -> Optional[Dict]:
        """Fit Gaussian to a single source."""
        t_map, q_map, u_map, weight = (frame["T"], frame["Q"], frame["U"], frame["Wpol"])
        maps.remove_weights(t_map, q_map, u_map, weight, zero_nans=False)

        if not check_zero_fraction(t_map, frame["Id"], max_zero_fraction=self.config.max_zero_fraction):
            return None

        fit = gaussfit_source(t_map, q_map, u_map, weight, config=self.config)
        yoff, xoff, t_amp, meanoff, q_amp, u_amp = fit

        if t_amp < self.config.min_t_amplitude:
            return None

        return {
            "yoff": yoff,
            "xoff": xoff,
            "t_amp": t_amp,
            "q_amp": q_amp,
            "u_amp": u_amp,
            "meanoff": meanoff,
            "maps": (t_map, q_map, u_map, weight),
        }

    def _extract_consistent_sources(self, gaussfit_results: Dict) -> Dict:
        """Extract sources that have data for all bands."""
        source_ids = self._find_common_sources(gaussfit_results)
        n_src = len(source_ids)
        n_bands = len(self.config.bands)
        ny, nx = self.map_shape

        arrays = {
            "gaussfit_yoff": np.zeros(n_src, dtype=self.config.dtype_np_real),
            "gaussfit_xoff": np.zeros(n_src, dtype=self.config.dtype_np_real),
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
            if not any(expected_id == sid.split("_")[0] for sid in results[band]):
                return False
        return True

    def _fill_source_arrays(self, arrays: Dict, results: Dict, source_ids: np.ndarray):
        """Fill numpy arrays with source data."""
        band_for_pos = "150GHz" if "150GHz" in self.config.bands else self.config.bands[0]

        for i, sid in enumerate(source_ids):
            pos_result = results[band_for_pos][f"{sid}-{band_for_pos}"]
            arrays["gaussfit_yoff"][i] = pos_result["yoff"]
            arrays["gaussfit_xoff"][i] = pos_result["xoff"]
            field_value = pos_result.get("field")
            if field_value is None:
                raise ValueError(f"Missing field metadata for source '{sid}'.")
            arrays["fields"][i] = field_value

            for j, band in enumerate(self.config.bands):
                result = results[band][f"{sid}-{band}"]
                self._fill_band_data(arrays, i, j, result)

    def _fill_band_data(self, arrays: Dict, src_idx: int, band_idx: int, result: Dict):
        """Fill arrays for a single band of a source."""
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

    def _create_leakage_template(self, gaussfit_amp: np.ndarray, raw_maps: np.ndarray, source_fields: np.ndarray):
        """Create or load the T->QU leakage correction template(s)."""
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

        return template, None

    def _load_precomputed_leakage_templates(
        self, source_fields: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Tuple[float, float]]]]:
        """Load precomputed leakage templates and stored source offsets."""
        unique_fields = {field for field in source_fields if field is not None}
        if not unique_fields:
            raise ValueError("No source fields available to match precomputed leakage templates.")

        templates = {}
        template_offsets: Dict[str, Dict[str, Tuple[float, float]]] = {}
        ny, nx = self.map_shape
        n_bands = len(self.config.bands)

        for field in unique_fields:
            field_templates = np.zeros((ny, nx, n_bands, 2), dtype=self.config.dtype_np_real)
            field_offsets: Dict[str, Tuple[float, float]] = template_offsets.setdefault(field, {})

            for band_idx, band in enumerate(self.config.bands):
                filename = os.path.join(
                    self.config.leakage_template_dir,
                    f"leakage_template_{field}_{band}_{self.config.leakage_weighting}.pkl",
                )

                if not os.path.exists(filename):
                    raise FileNotFoundError(
                        f"Missing leakage template for field '{field}', band '{band}', weighting '{self.config.leakage_weighting}': {filename}"
                    )

                with open(filename, "rb") as f:
                    template_data = pickle.load(f)

                try:
                    q_map = np.asarray(template_data["Q"], dtype=self.config.dtype_np_real)
                    u_map = np.asarray(template_data["U"], dtype=self.config.dtype_np_real)
                except (TypeError, KeyError) as err:
                    raise ValueError(f"Template file {filename} does not contain Q/U maps.") from err

                if q_map.shape != (ny, nx) or u_map.shape != (ny, nx):
                    raise ValueError(f"Template shape mismatch in {filename}: expected {(ny, nx)}, got {q_map.shape}/{u_map.shape}")

                field_templates[:, :, band_idx, 0] = q_map
                field_templates[:, :, band_idx, 1] = u_map

                offsets_payload = template_data.get("source_offsets")
                if offsets_payload is None:
                    raise ValueError(f"Template file {filename} does not contain 'source_offsets'.")

                for source_id, offsets in offsets_payload.items():
                    try:
                        y_offset = float(offsets["y_offset"])
                        x_offset = float(offsets["x_offset"])
                    except (KeyError, TypeError) as err:
                        raise ValueError(f"Invalid source offset format in {filename} for source '{source_id}'.") from err

                    # Always trust the offsets stored with each template; they supersede gaussfit positions.
                    field_offsets[source_id] = (y_offset, x_offset)

            templates[field] = field_templates

        return templates, template_offsets

    def _apply_template_offsets(self, source_data: Dict, template_offsets: Dict[str, Dict[str, Tuple[float, float]]]):
        """Overwrite initial offsets with the values stored alongside precomputed templates."""
        gaussfit_yoff = source_data["gaussfit_yoff"]
        gaussfit_xoff = source_data["gaussfit_xoff"]
        source_ids = source_data["source_ids"]
        source_fields = source_data["fields"]

        for idx, raw_source_id in enumerate(source_ids):
            sid = str(raw_source_id)
            field = source_fields[idx]
            field_offsets = template_offsets.get(field)
            if field_offsets is None or sid not in field_offsets:
                raise ValueError(f"Missing stored offset for source '{sid}' in field '{field}' while using precomputed templates.")

            y_offset, x_offset = field_offsets[sid]
            gaussfit_yoff[idx] = y_offset
            gaussfit_xoff[idx] = x_offset

    def _prepare_clean_maps(
        self,
        gaussfit_amp: np.ndarray,
        raw_maps: np.ndarray,
        qu_templates,
        source_fields: np.ndarray,
        gaussfit_yoff: np.ndarray,
        gaussfit_xoff: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply leakage correction, subtract shifted templates, and prepare final maps."""
        n_src, ny, nx, n_bands, _ = raw_maps.shape
        maps_clean = raw_maps.copy()

        for i in range(n_src):
            y_offset = float(gaussfit_yoff[i]) if gaussfit_yoff is not None else 0.0
            x_offset = float(gaussfit_xoff[i]) if gaussfit_xoff is not None else 0.0

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
                    if abs(y_offset) > 1e-6 or abs(x_offset) > 1e-6:
                        template_q_shifted = shift_map_bilinear(template_q, y_offset, x_offset)
                        template_u_shifted = shift_map_bilinear(template_u, y_offset, x_offset)
                    else:
                        template_q_shifted = template_q
                        template_u_shifted = template_u
                else:
                    template_q_shifted = template_q
                    template_u_shifted = template_u

                t_amp = gaussfit_amp[i, j, 0]
                maps_clean[i, :, :, j, 1] -= template_q_shifted * t_amp
                maps_clean[i, :, :, j, 2] -= template_u_shifted * t_amp

        maps_fft = None
        if self.config.chi2_method == "fourier":
            apod_mask = make_apodization_mask(self.map_shape, self.config.apodization_width_pix)
            apod_mask = apod_mask.astype(self.config.dtype_np_real)
            maps_fft = np.zeros((n_src, ny, nx, n_bands, 3), dtype=self.config.dtype_np_complex)

            for i in range(n_src):
                for j in range(n_bands):
                    apodized = maps_clean[i, :, :, j, :] * apod_mask[:, :, None]
                    maps_fft[i, :, :, j, :] = np.fft.fft2(apodized, axes=(0, 1))

        print(f"Prepared {n_src} sources for fitting.")
        return maps_clean, maps_fft
