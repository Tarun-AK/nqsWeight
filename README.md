# nqsWeight

## What this repo contains

This project is built around NetKet/JAX training code, custom variational-state reconstruction logic, model definitions, and data generation scripts.

## Main folders

- `models/` - model definitions:
  - `models/lstm.py`
  - `models/mpdoRnn.py`
  - `models/tensorNetwork.py`
  - `models/tensorNetworkOld.py`
  - `models/transformer.py`
  - `models/vanilla.py`
  - `models/__init__.py`
- `dataGeneration/` - dataset generation scripts and helpers:
  - `dataGeneration/Ising_GS_symmetric.jl`
  - `dataGeneration/anyonProbability.jl`
  - `dataGeneration/evolMPS.jl`
  - `dataGeneration/main`
  - `dataGeneration/sampling.jl`
  - `dataGeneration/toricGenerator.cpp`
  - `dataGeneration/toricGeneratorObc.cpp`
  - `dataGeneration/obcLoopContractibleV3.py`
  - `dataGeneration/obcLoopNoLogicalV3.py`

## Key files

- `regularizedQSR.py` - custom `QSR` training driver with a variance-reduction style objective.
- `toricTrainer.py` - training script for toric/surface-code style experiments.
- `isingTrainer.py` - training script for 1D Ising experiments.
- `weightStructure.py` - utilities for Walsh-Hadamard / Fourier-style analysis of log-probability structure.
- `cmi.py`, `klComp.py`, `toricPlots.py` - analysis helpers.

## Data generation

`dataGeneration/` contains a mix of Python, Julia, and C++ helpers. The upstream tree includes generators for periodic/open boundary loop data, toric-code variants, long-range models, and exact or sampling-based routines.

## Notes

- `logs/` is the main output location for training runs.
