# Repository guide

This guide maps the released files to the SASO workflow and records the assumptions needed to interpret them.

## Component map

### Formulation representation

`descriptor_utils.py` converts electrolyte records into four feature groups: the target ionic conductivity, continuous temperature and concentration features, categorical formulation features, and salt/solvent molecular descriptors. Its decoder reverses the numerical scaling. Molecular identities are recovered separately by `search_salt_solvent.py`.

### Generative models

- `model/MLPD`: multilayer-perceptron denoising diffusion.
- `model/R-MLPD`: diffusion with parallel routes and an input-dependent gate.
- `model/CVAE`: conditional variational autoencoder.
- `model/R-CVAE`: conditional variational autoencoder with routed decoders.

The released model definitions and training hyperparameters are retained unchanged. Maintenance changes should not modify them without a scientific reproducibility assessment.

### Molecular readout

`search_salt_solvent.py` maps generated descriptor vectors to discrete formulations. Salt matching uses Euclidean nearest neighbours. Binary-solvent matching enumerates unordered solvent pairs and 21 ratios between 0 and 1. A ternary-solvent function is included for exploratory use.

### Conductivity prediction

`prediction/route_model.py` defines the routed SCAN network. `prediction/feature.py` contains the released salt and solvent features, and `prediction/predict.py` illustrates inference. Five SCAN folds are stored under `trained_pth/`.

### Molecular-simulation helpers

- `scripts/box_construction.py` estimates molecule counts for a cubic simulation box.
- `scripts/conductivity_calculation.py` converts a Li+ diffusion coefficient to a Nernst-Einstein conductivity estimate.

These are research calculations rather than a validated general-purpose simulation package. Confirm unit conventions and assumptions before adapting them.

## Data contracts

### `data/training_data.csv`

| Column | Meaning |
|---|---|
| `k` | Ionic conductivity in mS/cm. |
| `T` | Temperature divided by 100 in the released table. |
| `c` | Salt concentration. |
| `salt` | Lithium-salt identifier. |
| `c units` | Concentration basis (`mol/kg` or `mol/l`). |
| `solvent ratio type` | Ratio basis (`mol`, `w`, or `v`). |
| `solvent_1` ... `solvent_4` | Solvent identifiers. |
| `ratio_1` ... `ratio_4` | Corresponding solvent fractions. |

`salt_MO.txt` and `solvent_MO.txt` contain two orbital values followed by the molecular identifier. `designed_formulation.csv` contains the 360 chemistry-guided formulations and their SCAN predictions.

## Reproducibility boundary

The release supports inspection of all model definitions, molecular matching, data audits, and access to pretrained weights. Several execution scripts retain working-environment conventions that are not fully represented in a clean checkout:

- training and generation scripts import `smiles.py`, whereas released descriptors are stored in text files and SCAN features are in `prediction/feature.py`;
- scripts refer to `compressed_new.csv`, while the released table is `data/training_data.csv`;
- generation and prediction scripts contain example checkpoint and input filenames;
- local imports require scripts to be run from their own directories unless deliberately adapted.

These differences are documented rather than automatically rewritten because an unverified compatibility layer could alter feature ordering, scaling, or checkpoint interpretation. A future reproducibility release should provide a canonical descriptor loader, configuration-driven paths, recorded random seeds and splits, and an end-to-end inference example checked against a reference output.

## Safe validation

These commands do not retrain models or overwrite artifacts:

```bash
python scripts/validate_repository.py
python -m unittest discover -s tests -v
```

The audit does not deserialize checkpoints. Model tests instantiate architectures with synthetic tensors only.
