import hashlib
import os
import pickle


class CacheManager:
    """Handles caching of the G3 file loading and T->P leakage subtraction."""

    def __init__(self, config):
        self.config = config
        os.makedirs(config.cache_dir, exist_ok=True)
        self.cache_filename = self._get_cache_filename()

    def _get_cache_filename(self):
        """Generates a unique filename based on data-loading config parameters."""
        # Select parameters that uniquely define the prepared data
        # if these parameters are different, it makes sense to start afresh
        # from the G3 files and re-calculate the leakage templates
        relevant_params = {
            "coadd_filenames": self.config.coadd_filenames,  # different input data
            "bands": self.config.bands,  # different data
            "map_size_pix": self.config.map_size_pix,  # different input data
            "reso_arcmin": self.config.reso_arcmin,  # different input data
            "min_t_amplitude": self.config.min_t_amplitude,  # different sources
            "max_zero_fraction": self.config.max_zero_fraction,  # different sources
            "skip_sources": sorted(self.config.skip_sources) if self.config.skip_sources else [],  # different sources
            "leakage_weighting": self.config.leakage_weighting,  # different approach to leakage
            "chi2_method": self.config.chi2_method,  # different chi2 calculation. if this is real_space, we don't need to take FFTs
        }
        # Create a stable string representation
        param_string = str(sorted(relevant_params.items()))
        # Hash the string to create a unique ID
        hasher = hashlib.md5()
        hasher.update(param_string.encode("utf-8"))
        return os.path.join(self.config.cache_dir, f"prepared_data_{hasher.hexdigest()}.pkl")

    def load(self):
        """Loads prepared data from cache if it exists."""
        if os.path.exists(self.cache_filename):
            print(f"Loading prepared data from cache: {self.cache_filename}")
            with open(self.cache_filename, "rb") as f:
                return pickle.load(f)
        return None

    def save(self, data):
        """Saves prepared data to cache."""
        print(f"Saving prepared data to cache: {self.cache_filename}")
        with open(self.cache_filename, "wb") as f:
            pickle.dump(data, f)

    def load_or_create(self, creation_func):
        """
        Tries to load data from cache. If it fails, it calls the
        creation_func to generate the data and then saves it to cache.
        """
        cached_data = self.load()
        if cached_data:
            return cached_data

        # Data not in cache, create it
        new_data = creation_func()
        self.save(new_data)
        return new_data
