# Polarized Beam Fitting

`polarized_beam_fitting` is a JAX-based analysis pipeline for measuring the polarized beam response of mm-wave telescopes using point-source observations. 

It was developed for SPT-3G, but the fitting code is experiment-agnostic everywhere except in the data-loading step. This package models temperature and polarization cutout maps, subtracts temperature-to-polarization leakage templates, and fits a shared beam model together with per-source position and flux parameters.

### SPT-3G Users

The default configuration in [`config.py`](config.py) points to local SPT-3G data products, cache directories, and betapol reference files used for *de Haan et al. (2026)*. To reproduce those results in another environment, override the paths and run settings before instantiating the fitter.

### Other Experiments

The SPT-3G-specific code is confined to the default data loader. To adapt the pipeline to another experiment:

1. Implement a loader subclass in the style of `ExampleExperimentDataLoader` in [`data_loader.py`](data_loader.py). Your loader should yield `SourceMapRecord` objects with T/Q/U maps in mK, weight maps in `1 / mK^2`, and pixel resolution in radians.
2. Use source IDs that are stable across bands. The fitter matches records by removing the configured band suffix, such as `-150GHz`.
3. Group input files by observing field in `config.coadd_filenames`, set the frequency bands in `config.bands`, and set `config.data_loader_class = YourExperimentDataLoader`.
4. Start with `config.use_precomputed_leakage_templates = False` unless you have generated leakage-template pickle files with the expected schema.
5. Choose a beam model. The `beta_pol` and `beta_T` models require a `betapol_data_path` file with two radial profiles to interpolate between; `gaussian` and B-spline-based models are easier starting points for new data.

## Paper

This code was used for the paper
"Characterization of the Polarization Beam Response of SPT-3G Using Point Sources"
by Tijmen de Haan, Melanie Archipley, Nicholas Huang, and the SPT-3G Collaboration. The paper was submitted in April 2026.

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

[`config.py`](config.py) defines a single mutable configuration object, `BeamFittingConfig`, with defaults for:

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

[`beam_model.py`](beam_model.py) provides a common interface for several beam parameterizations:

- `gaussian`: separate T and P Gaussian FWHM values
- `beta_pol`: fixed T beam from stitched profiles, polarized beam interpolated by `beta_pol`
- `beta_T`: test model that interpolates the T beam instead
- `bsplines_plus_gaussian`: shared Gaussian core plus orthonormal B-spline corrections for T and P
- `bsplines_plus_main`: fixed T beam plus B-spline perturbations around the polarization main beam

### Optimization and Sampling

[`fitter.py`](fitter.py) contains `PolarizedBeamFitter`, which:

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

[`bootstrap.py`](bootstrap.py) wraps the base fitter in `BootstrapBeamFitter`. It first finds the ML solution, then resamples sources with replacement.

## Unit Conventions

Unless a name or docstring explicitly states otherwise, map amplitudes and fluxes are in mK and angular quantities are in radians. 

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

Dependencies:

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

`spt3g_software` is optional. It is only required by the default loader for reading `.g3` coadd files. 

## File Guide

- [`__init__.py`](__init__.py): public exports
- [`config.py`](config.py): run configuration
- [`fitter.py`](fitter.py): ML fitting and posterior sampling
- [`beam_model.py`](beam_model.py): beam parameterizations
- [`data_loader.py`](data_loader.py): G3 loading, NumPy map preparation, and an example non-SPT data-loader adapter
- [`precision.py`](precision.py): Fourier covariance and precision construction
- [`bootstrap.py`](bootstrap.py): bootstrap resampling wrapper
- [`plotting.py`](plotting.py): diagnostics and summary figures
- [`template_construction.py`](template_construction.py): precompute leakage templates for iterative leakage handling
- [`source_fitting.py`](source_fitting.py): Gaussian source fits for initialization
- [`utils.py`](utils.py): interpolation, masks, parameter transforms, and other helpers
