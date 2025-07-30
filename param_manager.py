"""
Parameter management for polarized beam fitting.

This module provides a centralized ParameterManager class that handles all parameter
transformations, initialization, and organization using JAX pytrees.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map_with_path


def to_logit(value, bounds):
    """Transform value from physical space with bounds to unconstrained logit space."""
    low, high = bounds
    value_scaled = (value - low) / (high - low)
    # Clamp to avoid numerical issues
    value_scaled = jnp.clip(value_scaled, 1e-8, 1.0 - 1e-8)
    return jnp.log(value_scaled / (1.0 - value_scaled))


def from_logit(logit_value, bounds):
    """Transform value from unconstrained logit space back to physical space."""
    low, high = bounds
    return low + (high - low) * jax.nn.sigmoid(logit_value)


class ParameterManager:
    """A centralized handler for model parameters using JAX pytrees."""

    def __init__(self, config, beam_models, n_src):
        """
        Initialize the parameter manager.

        Parameters
        ----------
        config : BeamFittingConfig
            Configuration object with parameter specifications
        beam_models : dict
            Dictionary of beam models {band: BeamModel}
        n_src : int
            Number of sources
        """
        self.config = config
        self.beam_models = beam_models
        self.n_src = n_src
        self.n_bands = len(config.bands)
        self.bands = config.bands

        # Build parameter specifications
        self._spec_tree = self._build_spec_tree()

    def _build_spec_tree(self):
        """Build a pytree containing bounds and metadata for every parameter."""
        spec_tree = {"beams": {}, "sources": {}}

        # Beam specifications - get from each beam model
        for band in self.bands:
            # Wrap beam parameters in a "beam" key to match beam model expectations
            beam_spec = self._get_beam_spec(band)
            spec_tree["beams"][band] = {"beam": beam_spec}

        # Source specifications using flux_correction array
        spec_tree["sources"] = {
            "yoff": self.config.source_bounds[0],  # y offset bounds
            "xoff": self.config.source_bounds[1],  # x offset bounds
            "flux_correction": self.config.source_bounds[2],  # flux correction bounds (same for all stokes)
        }

        return spec_tree

    def _get_beam_spec(self, band):
        """Get parameter specifications for a specific beam model."""
        beam_model = self.beam_models[band]
        beam_spec = {}

        # Get bounds from config for each parameter the beam model defines
        for param_name in beam_model.param_names:
            beam_spec[param_name] = self.config.beam_coeff_bounds[param_name]

        return beam_spec

    def get_initial_params(self, initial_yoff=None, initial_xoff=None, in_logit_space=True):
        """
        Generate the initial parameter pytree.

        Parameters
        ----------
        initial_yoff : jax.Array, optional
            Initial y-offset values. If None, uses config default.
        initial_xoff : jax.Array, optional
            Initial x-offset values. If None, uses config default.
        in_logit_space : bool
            If True, return parameters in logit space for optimization.
            If False, return in physical space.

        Returns
        -------
        dict
            A pytree of JAX arrays containing all parameters
        """
        params_phys = {"beams": {}, "sources": {}}

        # Initialize beam parameters for each band
        for band in self.bands:
            # Wrap beam parameters in a "beam" key to match beam model expectations
            beam_params = self.beam_models[band].get_initial_physical_params()
            params_phys["beams"][band] = {"beam": beam_params}

        # Initialize source parameters using provided arrays or defaults
        if initial_yoff is not None:
            yoff_init = jnp.asarray(initial_yoff, dtype=jnp.float32)
        else:
            yoff_init = jnp.full(self.n_src, self.config.source_inits[0], dtype=jnp.float32)
            
        if initial_xoff is not None:
            xoff_init = jnp.asarray(initial_xoff, dtype=jnp.float32)
        else:
            xoff_init = jnp.full(self.n_src, self.config.source_inits[1], dtype=jnp.float32)

        params_phys["sources"] = {
            "yoff": yoff_init,
            "xoff": xoff_init,
            "flux_correction": jnp.full((self.n_src, self.n_bands, 3), self.config.source_inits[2], dtype=jnp.float32),
        }

        if in_logit_space:
            return self.to_logit(params_phys)
        return params_phys

    def to_physical(self, logit_params):
        """Convert an entire pytree of logit params to physical params."""
        return tree_map_with_path(lambda path, x: from_logit(x, self._get_bounds_from_path(path)), logit_params)

    def to_logit(self, physical_params):
        """Convert an entire pytree of physical params to logit params."""
        return tree_map_with_path(lambda path, x: to_logit(x, self._get_bounds_from_path(path)), physical_params)

    def _get_bounds_from_path(self, path):
        """Helper to retrieve bounds from the spec tree using a path from tree_map."""
        # Path is a tuple of keys, e.g., ('sources', 'yoff') or ('beams', '90GHz', 'beta_pol')
        current_level = self._spec_tree
        for key in path:
            current_level = current_level[key.key]
        return current_level

    def get_param_shapes(self):
        """Get the shapes of all parameters for debugging/validation."""
        initial_params = self.get_initial_params(in_logit_space=False)
        return jax.tree_util.tree_map(lambda x: x.shape, initial_params)

    def validate_params(self, params):
        """Validate that a parameter pytree has the correct structure and shapes."""
        expected_shapes = self.get_param_shapes()
        actual_shapes = jax.tree_util.tree_map(lambda x: x.shape, params)

        def check_shapes(expected, actual, path=""):
            if isinstance(expected, dict):
                if not isinstance(actual, dict):
                    raise ValueError(f"Expected dict at {path}, got {type(actual)}")
                for key in expected:
                    if key not in actual:
                        raise ValueError(f"Missing key '{key}' at {path}")
                    check_shapes(expected[key], actual[key], f"{path}.{key}")
            else:
                if expected != actual:
                    raise ValueError(f"Shape mismatch at {path}: expected {expected}, got {actual}")

        check_shapes(expected_shapes, actual_shapes)
        return True
