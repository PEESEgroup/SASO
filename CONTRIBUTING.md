# Contributing to SASO

Thank you for helping improve SASO. This repository is a research artifact, so changes should remain traceable to the accompanying scientific workflow.

## Development setup

```bash
conda env create -f environment.yml
conda activate saso
python scripts/validate_repository.py
python -m unittest discover -s tests -v
```

## Change guidelines

- Keep pull requests focused and explain the scientific or maintenance motivation.
- Do not change model architectures, feature definitions, preprocessing, training hyperparameters, or released checkpoints without documenting the effect on reported results.
- Add or update tests for reusable utility functions.
- Do not commit generated outputs, local environments, credentials, or proprietary data.
- Preserve the original data columns and clearly document any migration.
- Use explicit units in code, documentation, and result tables.

## Pull requests

A useful pull request description includes:

1. what changed;
2. why the change is needed;
3. whether scientific results or parameters are affected;
4. validation commands and their output;
5. any remaining reproducibility limitations.

Documentation-only and test-only improvements should explicitly state that no model code or parameters were changed.
