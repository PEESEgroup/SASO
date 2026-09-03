"""Run lightweight, non-mutating integrity checks for a SASO checkout."""

from __future__ import annotations

import ast
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FILES = (
    "README.md",
    "descriptor_utils.py",
    "search_salt_solvent.py",
    "data/training_data.csv",
    "data/designed_formulation.csv",
    "data/salt_MO.txt",
    "data/solvent_MO.txt",
    "trained_pt/MLPD.pth",
    "trained_pt/R-MLPD.pth",
    "trained_pt/CVAE.pt",
    "trained_pt/R-CVAE.pt",
)
TRAINING_COLUMNS = {
    "k", "T", "c", "salt", "c units", "solvent ratio type",
    "solvent_1", "ratio_1", "solvent_2", "ratio_2",
}


def read_header(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return set(next(csv.reader(handle)))


def count_descriptor_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> int:
    errors: list[str] = []
    for relative in EXPECTED_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    training_path = ROOT / "data" / "training_data.csv"
    if training_path.is_file():
        missing_columns = TRAINING_COLUMNS - read_header(training_path)
        if missing_columns:
            errors.append("training data is missing columns: " + ", ".join(sorted(missing_columns)))

    for relative, expected in {"data/salt_MO.txt": 13, "data/solvent_MO.txt": 38}.items():
        path = ROOT / relative
        if path.is_file():
            actual = count_descriptor_rows(path)
            if actual != expected:
                errors.append(f"{relative}: expected {expected} rows, found {actual}")

    scan_folds = sorted((ROOT / "trained_pth").glob("fold_*_model.pth"))
    if len(scan_folds) != 5:
        errors.append(f"expected 5 SCAN checkpoints, found {len(scan_folds)}")

    for path in ROOT.rglob("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            errors.append(f"cannot parse {path.relative_to(ROOT)}: {exc}")

    if errors:
        print("SASO repository validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("SASO repository validation passed.")
    print("  - required artifacts present")
    print("  - training schema present")
    print("  - 13 salt and 38 solvent descriptors present")
    print("  - five SCAN folds present")
    print("  - Python sources parse successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
