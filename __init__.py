"""
Polarized Beam Fitting Package

This package implements a maximum-likelihood forward-modeling approach to characterize
the polarized beam of SPT-3G using JAX optimization.

Author: Tijmen de Haan
Version: 2025-06-11
"""

from .fitter import PolarizedBeamFitter
from .bootstrap import BootstrapBeamFitter
from .config import BeamFittingConfig
from .plotting import BeamPlotter, create_diagnostic_plots
from .utils import (
    make_apodization_mask,
    make_apod_mask_center_excised,
    check_zero_fraction,
    compute_2d_asd,
    safe_filename,
)

from .source_fitting import gaussfit_source, fit_map_amplitude
from .beam_model import BeamModelBspline, BeamModelGaussian, BeamModelBetaPol, BeamModelBetaTest, BeamModelBSplinesGaussian
from .noise_psd import (
    NoisePSDCalculator,
    ClusterfinderPSDCalculator,
    KxAveragedIndividualCalculator,
    WhiteNoiseScaledCalculator,
    EnsembleAsdMeanCalculator,
    create_noise_psd_calculator,
)
from .param_manager import ParameterManager, to_logit, from_logit

__version__ = "1.0.0"
__author__ = "Tijmen de Haan"

__all__ = [
    "PolarizedBeamFitter",
    "BootstrapBeamFitter",
    "BeamFittingConfig",
    "BeamPlotter",
    "create_diagnostic_plots",
    "make_apodization_mask",
    "make_apod_mask_center_excised",
    "check_zero_fraction",
    "compute_2d_asd",
    "safe_filename",
    "gaussfit_source",
    "fit_map_amplitude",
    "BeamModelBspline",
    "BeamModelGaussian",
    "BeamModelBetaPol",
    "BeamModelBetaTest",
    "BeamModelBSplinesGaussian",
    "NoisePSDCalculator",
    "ClusterfinderPSDCalculator",
    "KxAveragedIndividualCalculator",
    "WhiteNoiseScaledCalculator",
    "EnsembleAsdMeanCalculator",
    "create_noise_psd_calculator",
    "ParameterManager",
    "to_logit",
    "from_logit",
]
