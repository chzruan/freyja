# Freyja MCMC Analysis Package

This package contains tools to perform a cosmological MCMC analysis using machine learning emulators for correlation functions ($\xi_0, \xi_2$) and the halo mass function (HMF).

<!-- ![image](https://github.com/chzruan/freyja/blob/main/dfo_freyja.png?raw=true) -->


## Installation

For research and development, it's best to install the package in "editable" mode. This means any changes you make to the source code will be immediately reflected when you run the code, without needing to reinstall.

Navigate to the root `freyja/` directory (the one containing `pyproject.toml`) and run:

```bash
pip install -e .
```

This will also install all the required dependencies like `jax`, `flax`, `numpyro`, etc.

## Usage

There are two primary ways to use this package.

### 1. As a Command-Line Tool

After installation, a command `freyja-run` becomes available. You can execute the entire MCMC pipeline by pointing it to your configuration file.

First, create a `config.yaml` file (you can copy the one from the original example). Then run:

```bash
freyja-run --config /path/to/your/config.yaml
```

### 2. As a Python Library

You can also import `freyja` into your own scripts or Jupyter notebooks to use its components programmatically. This is useful for debugging, plotting, or building more complex analyses.

Here is an example script:

```python
# example_script.py
from freyja.main import run_analysis

# Define the path to your configuration file
CONFIG_FILE = 'config.yaml'

print("Starting Freyja MCMC analysis from script...")
run_analysis(CONFIG_FILE)
print("Analysis complete.")
```