"""
Data loading and preprocessing for the polarized beam fitter.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from spt3g import core, maps

from .source_fitting import gaussfit_source
from .utils import check_zero_fraction, make_apodization_mask


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

        # Create leakage template
        qu_templates = self._create_leakage_template(source_data["gaussfit_amp"], source_data["raw_maps"])

        # Clean maps
        maps_clean, maps_fft = self._prepare_clean_maps(source_data["gaussfit_amp"], source_data["raw_maps"], qu_templates)

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
            source_data["n_src"],
        )

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

        for frame in g3_file:
            if frame.type != core.G3FrameType.Map or "Id" not in frame:
                continue

            source_id = frame["Id"]
            band = self._get_band_from_id(source_id)

            if band is None or self._should_skip_source(source_id):
                continue

            fit_result = self._fit_single_source(frame)
            if fit_result is not None:
                results[band][source_id] = fit_result

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

    def _create_leakage_template(self, gaussfit_amp: np.ndarray, raw_maps: np.ndarray) -> np.ndarray:
        """Create T->QU leakage correction template."""
        print(f"Creating T->P leakage template ({self.config.leakage_weighting} weighting)...")

        qu_maps = raw_maps[:, :, :, :, 1:3]
        t_amps = gaussfit_amp[:, :, 0]

        if self.config.leakage_weighting == "median":
            return np.median(qu_maps / t_amps[:, None, None, :, None], axis=0)

        weight_map = {"flat": 1.0, "linear": t_amps, "squared": t_amps**2}
        weights = weight_map[self.config.leakage_weighting][:, None, None, :, None]

        normalized_qu = qu_maps / t_amps[:, None, None, :, None]
        weighted_sum = np.sum(normalized_qu * weights, axis=0)
        weight_sum = np.sum(weights, axis=0)

        return weighted_sum / weight_sum

    def _prepare_clean_maps(self, gaussfit_amp: np.ndarray, raw_maps: np.ndarray, qu_templates: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply leakage correction and prepare final maps."""
        n_src, ny, nx, n_bands, _ = raw_maps.shape
        maps_clean = raw_maps.copy()

        for i in range(n_src):
            for j in range(n_bands):
                t_amp = gaussfit_amp[i, j, 0]
                maps_clean[i, :, :, j, 1] -= qu_templates[:, :, j, 0] * t_amp
                maps_clean[i, :, :, j, 2] -= qu_templates[:, :, j, 1] * t_amp

        maps_fft = None
        if self.config.chi2_method == "fourier":
            apod_mask = make_apodization_mask(self.map_shape, self.config.apodization_width_pix)
            maps_fft = np.zeros((n_src, ny, nx, n_bands, 3), dtype=self.config.dtype_np_complex)

            for i in range(n_src):
                for j in range(n_bands):
                    apodized = maps_clean[i, :, :, j, :] * apod_mask[:, :, None]
                    maps_fft[i, :, :, j, :] = np.fft.fft2(apodized, axes=(0, 1))

        print(f"Prepared {n_src} sources for fitting.")
        return maps_clean, maps_fft
