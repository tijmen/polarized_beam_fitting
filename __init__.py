"""
Polarized Beam Fitting Package

This package implements a maximum-likelihood forward-modeling approach to characterize
the polarized beam of SPT-3G using JAX optimization.

Author: Tijmen de Haan
Version: 2025-06-11
"""

from .beam_model import (
    BeamModelBetaPol,
    BeamModelBetaTest,
    BeamModelBSplinesGaussian,
    BeamModelGaussian,
)
from .bootstrap import BootstrapBeamFitter
from .config import BeamFittingConfig
from .fields import FieldCatalog
from .fitter import PolarizedBeamFitter
from .plotting import BeamPlotter, create_diagnostic_plots, create_mclmc_plots, create_nuts_plots
from .precision import calculate_precision
from .source_fitting import fit_map_amplitude, gaussfit_source
from .utils import (
    check_zero_fraction,
    from_logit,
    make_apod_mask_center_excised,
    make_apodization_mask,
    params_from_logit,
    params_to_logit,
    safe_filename,
    to_logit,
)

__version__ = "1.0.0"
__author__ = "Tijmen de Haan"

__all__ = [
    "PolarizedBeamFitter",
    "BootstrapBeamFitter",
    "BeamFittingConfig",
    "FieldCatalog",
    "BeamPlotter",
    "create_diagnostic_plots",
    "make_apodization_mask",
    "make_apod_mask_center_excised",
    "check_zero_fraction",
    "safe_filename",
    "to_logit",
    "from_logit",
    "params_to_logit",
    "params_from_logit",
    "gaussfit_source",
    "fit_map_amplitude",
    "BeamModelGaussian",
    "BeamModelBetaPol",
    "BeamModelBetaTest",
    "BeamModelBSplinesGaussian",
    "calculate_precision",
    "create_nuts_plots",
    "create_mclmc_plots",
]
