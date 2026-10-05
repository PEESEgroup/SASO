# Repository guide

This guide maps the released files to the SASO workflow and records the assumptions needed to interpret them.

## Component map

### Formulation representation

`descriptor_utils.py` separates the conductivity condition from the generated formulation. `ElectrolyteDataset.get_features_and_targets()` returns `(formulations, Y)`, where `Y[:, 0]` is the ionic conductivity `k` in `data/training_data.csv`. The formulation tensor contains temperature, concentration, categorical features, and salt/solvent molecular descriptors. Its decoder reverses the numerical scaling. Molecular identities are recovered separately by `search_salt_solvent.py`.

### Generative models

- `model/MLPD`: multilayer-perceptron denoising diffusion.
- `model/R-MLPD`: diffusion with parallel routes and an input-dependent gate.
- `model/CVAE`: conditional variational autoencoder.
- `model/R-CVAE`: conditional variational autoencoder with routed decoders.

Every training script exposes the loss variant, random seed, training budget, and principal architecture dimensions as command-line arguments. Standard and routed variants use the same shared loss utilities, so weighting is applied consistently to per-sample reconstruction losses before batch reduction. Routed branches have neutral names because specialization must be established from learned routing behaviour rather than inferred from labels.

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

The release supports inspection of model definitions, molecular matching, data audits, configurable training, generation, and access to pretrained weights. Training and generation use the canonical descriptor loader and `data/training_data.csv`; paths are resolved relative to the repository root. New checkpoints are accompanied by JSON provenance recording the architecture, objective, condition semantics, budget, seed, hyperparameters, and parameter count.

The standard and routed architectures do not have identical parameter counts. The six objective-routing ablations therefore isolate the effect of weighting within a fixed architecture and reveal its interaction with routing, but they should not be interpreted as a strict parameter-matched proof that every cross-architecture difference is caused solely by routing. Reproducing the released checkpoints also requires the software versions listed in `requirements.txt` and compatible hardware-dependent numerical libraries.

## Safe validation

These commands do not retrain models or overwrite artifacts:

```bash
python scripts/validate_repository.py
python -m unittest discover -s tests -v
```

The audit does not deserialize checkpoints. Model tests instantiate architectures with synthetic tensors only.
