# Polarized Beam Fitting

`polarized_beam_fitting` is a JAX-based fitter that fits a parametric radial beam profile to point source observations from mm-wave polarimeters such as SPT-3G. It models point-source cutout maps in temperature and polarization, subtracts T->Q/U leakage templates, and fits a shared beam model together with per-source position and flux parameters. It supports a number of beam models and minimizers/samplers.

### Note for SPT-3G Users

To anyone inside the SPT-3G collaboration who does want to reproduce the results from *de Haan et al. (2026)*, the default configuration in [`config.py`](/home/tijmen/cmb_analysis/beam_analysis/polarized_beam_fitting/config.py) points at SPT-3G data products, cache directories, and betapol reference files on my filesystem. Other users will need to override paths and run settings before using the package.

### Note for non-SPT Users

Despite this code only having been used for SPT-3G and simulated data so far, the SPT-3G specific code is limited entirely to the data loader. In order to run this pipeline with your own experiment, follow these steps:

1. Implement a loader subclass in the style of `ExampleExperimentDataLoader` in `data_loader.py`. Your loader should yield `SourceMapRecord` objects with T/Q/U maps in mK, weight maps in `1 / mK^2`, and pixel resolution in radians.
2. Specify the configuration, including grouping input files by observing field in `config.coadd_filenames`, setting observing bands in `config.bands`, and setting `config.data_loader_class = YourExperimentDataLoader`.
3. Choose a beam model. The `beta_pol` and `beta_T` models require a `betapol_data_path` file with two radial profiles to interpolate between; `gaussian` and B-spline-based models are easier starting points for new data.

## Paper

This code was used for the paper
"Characterization of the Polarization Beam Response of SPT-3G Using Point Sources"
by Tijmen de Haan, Melanie Archipley, Nicholas Huang, and the rest of the SPT-3G collaboration. The paper is in peer review as of April 3, 2026.

## What It Does

The main workflow is:

1. Read cutout map files for one or more observing fields and frequency bands.
2. Select sources that are present in every configured band and are not in the skip list.
3. Estimate or load per-source offsets and T/Q/U amplitudes.
4. Subtract a field-dependent T->Q/U leakage template, optionally using precomputed templates stored on disk.
5. Build either real-space weights or Fourier-space precision matrices.
6. Fit a beam model shared across sources while also fitting per-source offsets and fluxes.
7. Optionally run bootstrap resampling or MCMC posterior sampling for uncertainty estimation.

## Core Components

### Configuration

[`config.py`](/home/tijmen/cmb_analysis/beam_analysis/polarized_beam_fitting/config.py) defines a single mutable configuration object, `BeamFittingConfig`, with defaults for:

- input coadd files grouped by field
- cache and output directories
- selected bands
- map geometry and apodization
- beam model choice and parameter bounds
- skip-source handling
- leakage template mode
- CDRC calibration/deprojection parameters
- Fourier vs real-space chi-squared
- optimizer settings
- bootstrap and sampler settings

### Beam Models

[`beam_model.py`](/home/tijmen/cmb_analysis/beam_analysis/polarized_beam_fitting/beam_model.py) provides a common interface for several beam parameterizations:

- `gaussian`: separate T and P Gaussian FWHM values
- `beta_pol`: fixed T beam from stitched profiles, polarized beam interpolated by `beta_pol`
- `beta_T`: test model that interpolates the T beam instead
- `bsplines_plus_gaussian`: shared Gaussian core plus orthonormal B-spline corrections for T and P
- `bsplines_plus_main`: fixed T beam plus B-spline perturbations around the polarization main beam

### Optimization and Sampling

[`fitter.py`](/home/tijmen/cmb_analysis/beam_analysis/polarized_beam_fitting/fitter.py) contains `PolarizedBeamFitter`, which:

- builds the beam models for each configured band
- loads cached or freshly prepared source data
- constructs a real-space or Fourier-space objective
- parameterizes bounded parameters through logit transforms
- runs optimization with a tuned two-stage Adam/AMSGrad schedule, or alternative minimizers
- supports Hessian-based whitening for `NUTS`, `MCLMC`, and Newton-PCG

Useful methods:

- `run_fit()`: run maximum-likelihood optimization
- `sample_with_nuts()`: run posterior sampling with NumPyro NUTS
- `sample_with_mclmc()`: run BlackJAX MCLMC
- `create_model_maps()`: generate model thumbnails at fitted parameters
- `create_beam_profile_maps()`: generate centered T and P beam maps
- `calculate_individual_chi2s()`: inspect source-level fit quality

### Bootstrap Uncertainties

[`bootstrap.py`](/home/tijmen/cmb_analysis/beam_analysis/polarized_beam_fitting/bootstrap.py) wraps the base fitter in `BootstrapBeamFitter`. It first finds the ML solution, then resamples sources with replacement.

## Unit Conventions

Unless a name or docstring explicitly says otherwise, map amplitudes and fluxes are in mK and angular quantities are in radians. Configuration fields ending in `_arcmin` are stored in arcminutes and converted at the boundary where radian-valued math requires it.

## Typical Usage

This is the minimal package-level workflow:

```python
from polarized_beam_fitting import BeamFittingConfig, PolarizedBeamFitter, create_diagnostic_plots

config = BeamFittingConfig()

# Override site-specific defaults before running outside my environment.
config.coadd_filenames = {
    "myfield1": ["/path/to/myfield1_coadds.g3"],
}
config.output_dir = "/path/to/output"
config.cache_dir = "/path/to/cache"
config.leakage_template_dir = "/path/to/cache/leakage_templates"
config.betapol_data_path = "/path/to/polarized_beam_fitting/data/betapol_TdH.npz"

fitter = PolarizedBeamFitter(config)
best_fit_params = fitter.run_fit()
print(best_fit_params["beams"])

create_diagnostic_plots(fitter, best_fit_params)
```

## Runtime Expectations

This code frankly has too many dependencies. Important are:

- `jax`
- `numpy`
- `scipy`
- `matplotlib`
- `optax`
- `optimistix`
- `blackjax`
- `numpyro`
- `camb`
- `arviz`
- `corner`

`spt3g_software` is optional and only required for reading `.g3` coadd files. The data loader converts G3 containers into plain NumPy arrays before handing data to the rest of the package.

## File Guide

- [`__init__.py`](/home/tijmen/cmb_analysis/beam_analysis/polarized_beam_fitting/__init__.py): public exports
- [`config.py`](/home/tijmen/cmb_analysis/beam_analysis/polarized_beam_fitting/config.py): run configuration
- [`fitter.py`](/home/tijmen/cmb_analysis/beam_analysis/polarized_beam_fitting/fitter.py): ML fitting and posterior sampling
- [`beam_model.py`](/home/tijmen/cmb_analysis/beam_analysis/polarized_beam_fitting/beam_model.py): beam parameterizations
- [`data_loader.py`](/home/tijmen/cmb_analysis/beam_analysis/polarized_beam_fitting/data_loader.py): G3 loading, NumPy map preparation, and an example non-SPT data-loader adapter
- [`precision.py`](/home/tijmen/cmb_analysis/beam_analysis/polarized_beam_fitting/precision.py): Fourier covariance and precision construction
- [`bootstrap.py`](/home/tijmen/cmb_analysis/beam_analysis/polarized_beam_fitting/bootstrap.py): bootstrap resampling wrapper
- [`plotting.py`](/home/tijmen/cmb_analysis/beam_analysis/polarized_beam_fitting/plotting.py): diagnostics and summary figures
- [`template_construction.py`](/home/tijmen/cmb_analysis/beam_analysis/polarized_beam_fitting/template_construction.py): precompute leakage templates for iterative leakage handling
- [`source_fitting.py`](/home/tijmen/cmb_analysis/beam_analysis/polarized_beam_fitting/source_fitting.py): Gaussian source fits for initialization
- [`utils.py`](/home/tijmen/cmb_analysis/beam_analysis/polarized_beam_fitting/utils.py): interpolation, masks, parameter transforms, and other helpers
