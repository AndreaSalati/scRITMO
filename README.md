# scRITMO
A suite of useful functions for single cell analysis, pseudobulk and rhythmic analysis.  

All probabilistic models based on jax are not included yet in the package, but the scripts are in the repo.  

## Installation

### Basic Installation (core dependencies only)

```bash
# Clone the repository
git clone https://github.com/AndreaSalati/scRITMO.git
cd scRITMO

# Create a conda environment with Python 3.11
conda create -n scritmo-env python=3.11 -y
conda activate scritmo-env

# Install the package
pip install -e .
```

### Installation with ML extras (includes PyTorch)

For full functionality including the `scritmo.ml` module:

```bash
# Clone the repository
git clone https://github.com/AndreaSalati/scRITMO.git
cd scRITMO

# Create a conda environment with Python 3.11
conda create -n scritmo-env python=3.11 -y
conda activate scritmo-env

# Install the package with ML extras
pip install -e ".[ml]"
```

### Verify Installation

```bash
python -c "import scritmo; print('✓ scritmo installed successfully')"
python -c "from scritmo import ml; print('✓ scritmo.ml installed successfully')"
```

## Requirements

- Python >= 3.11
- See `pyproject.toml` for full dependency list

## Optional Dependencies

- `[ml]` extras include: `torch>=2.0.0`, `torchvision>=0.15.0`
