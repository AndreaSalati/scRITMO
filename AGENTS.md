# scRITMO - Agent Documentation

## Environment

Use the **ML-gpu** conda environment for all development and testing:
```bash
conda activate ML-gpu
# or: conda run -n ML-gpu python ...
```
This env has Python 3.12.6 and all dependencies pre-installed.

## Project Overview

**scRITMO** is a Python package for single-cell rhythmic analysis, providing tools for analyzing circadian rhythms in single-cell RNA sequencing data. The package focuses on:

- **Rhythmic gene expression analysis** using harmonic regression
- **Pseudobulk aggregation** of single-cell data
- **Probabilistic modeling** with JAX/NumPyro (RITMO models)
- **Circular statistics** for phase-based analysis
- **Phase inference algorithms** (CHIRAL - spin glass + EM algorithm)
- **Specialized RNA-seq analysis** (DrySeq pipeline)

The project targets Python 3.11+ and integrates with the standard single-cell analysis ecosystem (Scanpy, AnnData).

## Technology Stack

- **Language**: Python >=3.11
- **Build System**: setuptools (configured in `pyproject.toml`)
- **Core Dependencies**:
  - Single-cell: `scanpy>=1.10.0`, `anndata>=0.12.7`
  - Numerical: `numpy>=1.26.0`, `scipy>=1.11.0`, `pandas>=2.1.0`
  - Statistics: `statsmodels>=0.14.1`, `patsy>=1.0.1`
  - ML: `scikit-learn>=1.4.0`, `joblib>=1.4.2`
  - Visualization: `matplotlib>=3.7.1`, `seaborn>=0.13.0`, `adjustText>=1.3.0`
  - Probabilistic (optional): JAX, NumPyro (for `jax_module`)

## Project Structure

```
scRITMO/
├── pyproject.toml              # Package configuration
├── README.md                   # Brief project description
├── src/
│   └── scritmo/               # Main package source
│       ├── __init__.py        # Package exports
│       ├── basics.py          # Core utilities (BIC, indexing, conversions)
│       ├── beta.py            # Beta coefficient DataFrame class for harmonics
│       ├── gene_lists.py      # Core clock genes (human, drosophila)
│       ├── glm.py             # GLM fitting (Negative Binomial, Poisson, Gaussian)
│       ├── linear_regression.py  # Harmonic regression utilities
│       ├── ppca.py            # Probabilistic PCA
│       ├── pseudobulk.py      # Pseudobulk aggregation functions
│       ├── power_spherical.py # Power spherical distribution utilities
│       ├── circular/          # Circular statistics submodule
│       │   ├── align.py       # Phase alignment utilities
│       │   ├── circular.py    # Circular deviation metrics
│       │   ├── median.py      # Circular median calculations
│       │   └── von_mises.py   # Von Mises distribution functions
│       ├── dryseq/            # DrySeq analysis pipeline
│       │   ├── dryseq_main.py # Main DrySeq execution
│       │   ├── dryseq_fitting.py  # Iterative model fitting
│       │   └── dryseq_utils.py    # Design matrix utilities
│       ├── jax_module/        # JAX/NumPyro probabilistic models
│       │   ├── RITMO.py       # Main RITMO class for inference
│       │   ├── RITMO_base.py  # DataLoader base class
│       │   ├── RITMO_helper.py # Helper functions for RITMO
│       │   ├── numpyro_models.py # Core NumPyro model definitions
│       │   ├── numpyro_models_handles.py # Model handlers
│       │   ├── posterior.py   # Posterior analysis utilities
│       │   └── simulations.py # Dataset simulation
│       ├── plot/              # Visualization submodule
│       │   ├── bar_box.py     # Bar and box plots
│       │   ├── cedric_fn.py   # Specialized plotting functions
│       │   ├── data_and_fits.py # Data visualization with fits
│       │   ├── histos.py      # Histogram utilities
│       │   ├── misc.py        # Miscellaneous plots
│       │   ├── panels_constants.py # Panel plot constants
│       │   ├── panels_plot_helpers.py # Panel plot helpers
│       │   ├── plot_generator_fns.py # Plot generation utilities
│       │   └── utils.py       # Plotting utilities (polar plots, etc.)
│       └── pychiral/          # CHIRAL phase inference algorithm
│           ├── chiral.py      # Main CHIRAL implementation
│           ├── em.py          # Expectation-Maximization steps
│           ├── helper_fn.py   # Helper functions
│           └── stat_phys.py   # Statistical physics (spin glass)
├── dist/                      # Built distribution files (wheel, tar.gz)
└── __marimo__/                # Marimo notebook cache (empty)
```

## Build and Installation

### Development Installation
```bash
git clone https://github.com/AndreaSalati/scRITMO.git
pip install -e ./scRITMO
```

### Building Distribution
```bash
# Build wheel and source distribution
python -m build

# Output goes to dist/:
# - scritmo-0.1.0-py3-none-any.whl
# - scritmo-0.1.0.tar.gz
```

## Module Reference

### Core Utilities (`basics.py`)
- `LL()`, `BIC()`: Log-likelihood and Bayesian Information Criterion
- `ind()`, `ind2()`: Index lookup utilities
- `dict2df()`, `df2dict()`: Dictionary/DataFrame conversions
- `fold_change()`: Log-amplitude to fold-change conversion
- `w`, `rh`: Global constants (angular frequency = 2π/24, reciprocal)

### Beta Coefficients (`beta.py`)
- `Beta` class: Pandas DataFrame subclass for harmonic coefficients
  - Handles columns named `a_0`, `a_1`, `b_1`, `a_2`, `b_2`, etc.
  - Methods for extracting amplitude/phase, polar plotting
- `cSVD_beta()`: Circular SVD for beta coefficients
- `plot_beta_shift()`: Visualization for beta shifts

### Pseudobulk (`pseudobulk.py`)
- `pseudobulk()`: Create pseudobulk AnnData with optional pseudo-replicates
- `pseudo_bulk_time()`: Map sample labels to time values
- `normalize_log_PB()`: Normalize and log-transform pseudobulk data

### GLM Fitting (`glm.py`)
- `glm_gene_fit()`: Fit GLM models per gene with multiple noise models:
  - Negative Binomial (`nb`)
  - Poisson (`poisson`)  
  - Gaussian (`gaussian`)
- Supports parallel processing via joblib
- Includes outlier filtering and likelihood ratio testing

### Linear Regression (`linear_regression.py`)
- `create_harmonic_design_matrix()`: Build design matrices for harmonic regression
- `evaluate_harmonic_fn()`: Evaluate harmonic functions from coefficients
- `fit_periodic_spline()`: Fit periodic splines
- `harmonic_regression_loop()`: Batch harmonic regression on AnnData
- `polar_genes_pandas()`: Polar coordinates for gene phases/amplitudes

### Circular Statistics (`circular/`)
- `circular_deviation()`, `circular_square_error()`: Error metrics for circular data
- `mean_AE()`, `median_AE()`: Mean/median absolute error for circular data
- `optimal_shift()`, `get_shift_y()`: Phase alignment utilities
- Von Mises distribution functions

### RITMO (`jax_module/`)
Probabilistic modeling using JAX/NumPyro:
- `RITMO` class: Main interface for rhythmic inference
- `model_MLE_NB`: Negative Binomial MLE model
- `model_MLE_G`: Gaussian MLE model
- `model_null`: Null model for hypothesis testing
- Supports GPU acceleration via `numpyro.set_platform("gpu")`

### CHIRAL (`pychiral/`)
Phase inference algorithm combining:
- Spin glass initialization (mean field)
- Expectation-Maximization for refinement
- Two-state model support (TSM)

### DrySeq (`dryseq/`)
Pipeline for rhythmic bulk RNA-seq analysis:
- Dispersion estimation via PyDESeq2
- Multiple rhythmic model comparison using BIC weights
- Parallel model fitting

### Plotting (`plot/`)
Extensive visualization utilities:
- `polar_plot()`: Configurable polar axes for phase data
- `xy()`: Diagonal reference lines
- Gene expression heatmaps, histograms, bar/box plots
- Panel plot layouts for multi-gene visualization

### Gene Lists (`gene_lists.py`)
Predefined clock gene sets:
- `hccg`: Human core clock genes
- `hccg_extended`: Extended human clock gene set
- `dccg`: Drosophila core clock genes

## Development Conventions

### Code Style
- **Docstrings**: Google-style docstrings with type hints
- **Type hints**: Used in function signatures (e.g., `Iterable[Hashable]`, `AnnData`)
- **Naming**: 
  - Functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE` (global) or module-level
- **Global constants** in `basics.py`: `w` (angular frequency), `rh` (reciprocal)

### Data Conventions
- **Phases**: Expressed in radians (0 to 2π)
- **Time**: Typically "ZT" (Zeitgeber Time) in hours, 24-hour period
- **AnnData**: Standard Scanpy AnnData objects with:
  - `.X` or layers (e.g., `"spliced"`) for counts
  - `.obs` metadata including time information

### Key Patterns
1. **Gene filtering**: Most methods filter out genes with zero expression
2. **Pseudobulk workflow**: Aggregate cells → normalize → log-transform → analyze
3. **Harmonic regression**: Design matrix → fit → extract amplitude/phase
4. **Model comparison**: Fit alternative and null models → likelihood ratio test

## Testing

**No formal test suite is currently implemented.** The project lacks:
- No `tests/` directory
- No `pytest.ini` or test configuration
- No CI/CD pipelines

Testing appears to be done manually or through example notebooks.

## Dependencies Notes

### Optional Dependencies
The JAX/NumPyro modules require additional dependencies not listed in `pyproject.toml`:
- `jax`
- `numpyro`
- `pydeseq2` (for DrySeq module)

These are commented out in `__init__.py` and must be installed separately:
```python
# try:
#     import jax
#     import numpyro
#     from .jax_module import *
# except Exception as e:
#     pass
```

## Version

Current version: **0.1.0**

## License

MIT License (referenced in `pyproject.toml`, no LICENSE file present)

## Author

Andrea Salati (<andrea.salati96@gmail.com>)

## Notes for AI Agents

1. **Modifying models**: When editing JAX/NumPyro models, be aware of the handler pattern in `numpyro_models_handles.py`
2. **Adding genes**: Use `gene_lists.py` to add new predefined gene sets
3. **Circular data**: Always use functions from `circular/` module for phase arithmetic, never use linear operations
4. **Beta coefficients**: The `Beta` class maintains column ordering `a_0, a_1, b_1, a_2, b_2, ...`
5. **GPU usage**: RITMO models default to GPU; set `jax_device="cpu"` for CPU-only execution
6. **AnnData layers**: The package frequently uses layers like `"spliced"`; check layer existence before operations
