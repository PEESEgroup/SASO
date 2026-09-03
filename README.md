# SASO

**Data-aware conditional generation for non-aqueous electrolyte formulations**

SASO is a research codebase for generating salt-solvent formulations conditioned on ionic conductivity. It combines molecular-orbital descriptors, conditional generative models, molecular readout, and the SCAN surrogate model in a generation-to-screening workflow.

<p align="center">
  <img width="900" alt="Overview of the SASO workflow" src="https://github.com/user-attachments/assets/a01f1cfa-4d99-478c-a543-28a6ff6dccba" />
</p>

> [!IMPORTANT]
> This repository accompanies a research manuscript and is intended for research use. Generated or predicted formulations require independent computational and experimental validation before use.

## Overview

SASO provides four model variants for property-conditioned formulation generation:

| Model | Generative formulation | Dynamic routing |
|---|---|---:|
| MLPD | Multilayer-perceptron diffusion | No |
| R-MLPD | Multilayer-perceptron diffusion | Yes |
| CVAE | Conditional variational autoencoder | No |
| R-CVAE | Conditional variational autoencoder | Yes |

The repository also includes:

- HOMO-LUMO descriptors for 13 lithium salts and 38 solvents;
- the CALiSol-derived training table used in the study;
- pretrained weights for the four generators and five SCAN folds;
- nearest-neighbour molecular-orbital readout utilities;
- scripts for simulation-box composition and Nernst-Einstein conductivity calculation;
- lightweight repository and unit tests.

## Workflow

```text
training data + molecular-orbital descriptors
                    |
                    v
       conditional generative models
        (MLPD / R-MLPD / CVAE / R-CVAE)
                    |
                    v
       continuous formulation vectors
                    |
                    v
     salt and solvent nearest-neighbour readout
                    |
                    v
           SCAN surrogate ranking
                    |
                    v
       molecular simulation / experiment
```

## Repository layout

```text
SASO/
├── data/                       # Training data, designed formulations, MO descriptors
├── model/
│   ├── MLPD/                   # Baseline diffusion model and training script
│   ├── R-MLPD/                 # Routing-enabled diffusion model
│   ├── CVAE/                   # Baseline conditional VAE
│   └── R-CVAE/                 # Routing-enabled conditional VAE
├── prediction/                 # SCAN model, features, and prediction workflow
├── scripts/                    # Molecular-simulation helper calculations
├── tests/                      # Unit and repository-integrity tests
├── trained_pt/                 # Pretrained generative-model weights
├── trained_pth/                # Five pretrained SCAN folds
├── descriptor_utils.py         # Dataset encoding and decoding
├── search_salt_solvent.py      # MO-space molecular readout
└── generate.py                 # Original generation workflow snapshot
```

See [`docs/REPOSITORY_GUIDE.md`](docs/REPOSITORY_GUIDE.md) for a component-level description and the current reproducibility boundary.

## Installation

### Conda

```bash
git clone https://github.com/PEESEgroup/SASO.git
cd SASO
conda env create -f environment.yml
conda activate saso
```

### pip

Create a Python 3.11 environment, then run:

```bash
python -m pip install -r requirements.txt
```

The recorded research environment used PyTorch 2.4.1, pandas 2.2.2, and RDKit 2024.3.6. Exact GPU installation commands may vary with the local CUDA version; consult the PyTorch installation selector when GPU support is required.

## Data

| File | Description |
|---|---|
| `data/training_data.csv` | 13,302 electrolyte records used by the formulation workflow. Temperature is stored after division by 100, following the original preprocessing. |
| `data/salt_MO.txt` | HOMO-LUMO descriptors for lithium salts. |
| `data/solvent_MO.txt` | HOMO-LUMO descriptors for solvents. |
| `data/designed_formulation.csv` | 360 chemistry-guided candidate formulations evaluated with SCAN. |

Column names and units are documented in [`docs/REPOSITORY_GUIDE.md`](docs/REPOSITORY_GUIDE.md#data-contracts).

## Usage

### Validate a checkout

Run the dependency-free repository audit first:

```bash
python scripts/validate_repository.py
```

It verifies expected files, data schemas, descriptor counts, checkpoint inventory, and Python syntax without loading model weights.

### Run the tests

```bash
python -m unittest discover -s tests -v
```

Model-interface tests are skipped automatically when PyTorch is unavailable. The tests do not train models or modify checkpoints.

### Train a model

Each training script is colocated with its architecture. Run from that model directory because the original scripts use working-directory-relative paths:

```bash
cd model/R-MLPD
python train.py
```

The scripts preserve the hyperparameters used in the research workflow. Before retraining, see the reproducibility note below about descriptor-module and training-table filenames expected by the original scripts.

### Predict conductivity with SCAN

The prediction implementation is in `prediction/`. It consumes salt, solvent, and condition feature blocks and loads a SCAN checkpoint:

```bash
cd prediction
python predict.py
```

Update the input CSV and checkpoint paths in the research script for the formulation set being evaluated.

### Molecular-orbital readout

`search_salt_solvent.py` exposes reusable nearest-neighbour functions:

```python
import numpy as np

from search_salt_solvent import search_best_2solvent

solvents = {
    "solvent_a": np.array([-8.0, 1.0]),
    "solvent_b": np.array([-7.0, 0.5]),
}

match, distance = search_best_2solvent(
    target_vec=np.array([-7.5, 0.75]),
    solvent_features_dict=solvents,
)
```

### Simulation helpers

```bash
python scripts/box_construction.py
python scripts/conductivity_calculation.py
```

These scripts contain the calculation examples used in the study. Check all units and physical assumptions before adapting them to a new system.

## Reproducibility status

The repository contains the manuscript datasets, model definitions, and pretrained weights. The original training and generation scripts also refer to historical local filenames such as `compressed_new.csv`, `smiles.py`, and model-specific checkpoint names. These names are retained to avoid silently changing the published research pipeline. A clean checkout therefore supports code inspection, repository validation, molecular readout tests, and access to the released artifacts, but some end-to-end commands require the authors to reconcile those historical inputs with the released `data/` files.

This boundary is recorded explicitly in [`docs/REPOSITORY_GUIDE.md`](docs/REPOSITORY_GUIDE.md#reproducibility-boundary). Contributions that make the workflow portable are welcome provided they preserve the reported model architectures and parameters.

## Contributing

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) before proposing changes. For scientific changes, document whether model architecture, preprocessing, hyperparameters, or numerical results are affected.

## Citation

If SASO supports your research, please cite:

```text
Zhilong Wang, Wentao Hou, Xianyong Wu, and Fengqi You. Manuscript submitted (2026).
```

The citation will be updated when the article record becomes available.

## Authors

The software was primarily developed by **Dr. Zhilong Wang**, advised by **Prof. Fengqi You**, with research contributions described in the accompanying manuscript.

## License

This software is under the MIT License.
