# Funnels

This repository contains the code for the experiments presented in 

> Funneling flows. Exact maximum likelihood with dimensionality reduction.

## Dependencies
See `setup.py` for necessary pip packages.

Tested with Python 3.8.5 and PyTorch 1.1.1

## Usage

The path variables in `surVAE/utils/io.py` need to be set before running experiments.

#### Plane experiments
Use `experiments/plane_data_generation.py`

#### UCI and BSDS experiments
Use `experiments/uci.py`

#### Image generation
Use `experiments/no_recon/image_generation.py` 
