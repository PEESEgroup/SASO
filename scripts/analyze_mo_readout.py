from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


GRID_STEP = 0.05
EPS = 1e-12



SOLVENT_ALIASES = {
    "3-Me-2-O": "3-Me-2-Oxazolidinone",
    "3-MeSul": "3-MeSulfolane",
    "Ethylb": "Ethylbenzene",
    "Ethyld": "Ethyldiglyme",
    "Ethylm": "Ethylmonoglyme",
    "Freon": "Freon 11",
    "MC": "Methylene chloride",
    "g-Buty": "g-Butyrolactone",
}


def load_descriptors(
    path: Path, aliases: dict[str, str] | None = None
) -> dict[str, np.ndarray]:
    descriptors: dict[str, np.ndarray] = {}
    text = path.read_text(encoding="utf-8")
    dictionary_entries = re.findall(
        r"['\"]([^'\"]+)['\"]\s*:\s*np\.array\(\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]\)",
        text,
    )
    if dictionary_entries:
        for name, first, second in dictionary_entries:
            if aliases:
                name = aliases.get(name, name)
            descriptors[name] = np.asarray([first, second], dtype=float)
        return descriptors
    for line in text.splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        name = " ".join(fields[2:])
        if aliases:
            name = aliases.get(name, name)
        descriptors[name] = np.asarray(fields[:2], dtype=float)
    return descriptors


def make_solvent_candidates(
    descriptors: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[tuple[tuple[str, float], ...]]]:
    names = sorted(descriptors)
    vectors: list[np.ndarray] = []
    compositions: list[tuple[tuple[str, float], ...]] = []

    for name in names:
        vectors.append(descriptors[name])
        compositions.append(((name, 1.0),))

    grid = np.arange(GRID_STEP, 1.0, GRID_STEP)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            for fraction_a in grid:
                fraction_b = 1.0 - fraction_a
                vectors.append(
                    fraction_a * descriptors[name_a]
                    + fraction_b * descriptors[name_b]
                )
                compositions.append(
                    ((name_a, float(fraction_a)), (name_b, float(fraction_b)))
                )

    return np.vstack(vectors), compositions


def nearest_two(
    targets: np.ndarray, candidates: np.ndarray, chunk_size: int = 256
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate_norm = np.einsum("ij,ij->i", candidates, candidates)
    first_idx = np.empty(len(targets), dtype=int)
    second_idx = np.empty(len(targets), dtype=int)
    first_dist = np.empty(len(targets), dtype=float)
    second_dist = np.empty(len(targets), dtype=float)

    for start in range(0, len(targets), chunk_size):
        stop = min(start + chunk_size, len(targets))
        block = targets[start:stop]
        distances_sq = (
            np.einsum("ij,ij->i", block, block)[:, None]
            + candidate_norm[None, :]
            - 2.0 * block @ candidates.T
        )
        np.maximum(distances_sq, 0.0, out=distances_sq)
        pair = np.argpartition(distances_sq, kth=1, axis=1)[:, :2]
        pair_dist = np.take_along_axis(distances_sq, pair, axis=1)
        order = np.argsort(pair_dist, axis=1)
        pair = np.take_along_axis(pair, order, axis=1)
        pair_dist = np.take_along_axis(pair_dist, order, axis=1)
        first_idx[start:stop] = pair[:, 0]
        second_idx[start:stop] = pair[:, 1]
        first_dist[start:stop] = np.sqrt(pair_dist[:, 0])
        second_dist[start:stop] = np.sqrt(pair_dist[:, 1])

    return first_idx, second_idx, first_dist, second_dist


def row_composition(row: pd.Series) -> tuple[tuple[str, float], ...]:
    values = []
    for i in range(1, 5):
        name = row.get(f"solvent_{i}")
        ratio = row.get(f"ratio_{i}")
        if pd.notna(name) and pd.notna(ratio) and float(ratio) > EPS:
            values.append((str(name), float(ratio)))
    return tuple(sorted(values))


def composition_dict(
    composition: tuple[tuple[str, float], ...], normalize: bool = False
) -> dict[str, float]:
    values = {name: float(ratio) for name, ratio in composition}
    if normalize:
        total = sum(values.values())
        if total > 0:
            values = {name: ratio / total for name, ratio in values.items()}
    return values


def composition_errors(
    actual: tuple[tuple[str, float], ...],
    predicted: tuple[tuple[str, float], ...],
) -> dict[str, float | bool]:
    actual_raw = composition_dict(actual)
    predicted_raw = composition_dict(predicted)
    actual_norm = composition_dict(actual, normalize=True)
    predicted_norm = composition_dict(predicted, normalize=True)
    names = sorted(set(actual_raw) | set(predicted_raw))
    identity_match = set(actual_raw) == set(predicted_raw)
    raw_abs = [abs(actual_raw.get(n, 0.0) - predicted_raw.get(n, 0.0)) for n in names]
    norm_abs = [abs(actual_norm.get(n, 0.0) - predicted_norm.get(n, 0.0)) for n in names]
    max_raw = max(raw_abs, default=0.0)
    max_norm = max(norm_abs, default=0.0)
    return {
        "identity_match": identity_match,
        "ratio_l1_error_raw": float(sum(raw_abs)),
        "ratio_max_abs_error_raw": float(max_raw),
        "ratio_l1_error_normalized": float(sum(norm_abs)),
        "ratio_max_abs_error_normalized": float(max_norm),
        "strict_composition_match": bool(identity_match and max_raw <= 1e-10),
        "composition_match_tol_0_025": bool(identity_match and max_norm <= 0.025 + EPS),
        "composition_match_tol_0_05": bool(identity_match and max_norm <= 0.05 + EPS),
    }


def format_composition(composition: tuple[tuple[str, float], ...]) -> str:
    return " + ".join(f"{name}:{ratio:.6g}" for name, ratio in composition)


def build_known_targets(
    data: pd.DataFrame, descriptors: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, np.ndarray, list[tuple[tuple[str, float], ...]]]:
    component_count = data[[f"solvent_{i}" for i in range(1, 5)]].notna().sum(axis=1)
    subset = data.loc[component_count <= 2].copy()
    subset.insert(0, "source_row", subset.index + 2)
    compositions = [row_composition(row) for _, row in subset.iterrows()]
    targets = np.vstack(
        [
            sum((ratio * descriptors[name] for name, ratio in comp), np.zeros(2))
            for comp in compositions
        ]
    )
    return subset, targets, compositions


def salt_recovery(
    data: pd.DataFrame, descriptors: dict[str, np.ndarray]
) -> pd.DataFrame:
    names = sorted(descriptors)
    vectors = np.vstack([descriptors[name] for name in names])
    targets = np.vstack([descriptors[name] for name in data["salt"]])
    i1, i2, d1, d2 = nearest_two(targets, vectors, chunk_size=1024)
    return pd.DataFrame(
        {
            "source_row": data.index + 2,
            "actual_salt": data["salt"].astype(str),
            "predicted_salt": [names[i] for i in i1],
            "salt_identity_match": [names[i] == actual for i, actual in zip(i1, data["salt"])],
            "salt_descriptor_distance": d1,
            "salt_second_candidate": [names[i] for i in i2],
            "salt_second_distance": d2,
            "salt_margin": d2 - d1,
        }
    )


def solvent_recovery(
    data: pd.DataFrame,
    descriptors: dict[str, np.ndarray],
    candidates: np.ndarray,
    candidate_compositions: list[tuple[tuple[str, float], ...]],
) -> pd.DataFrame:
    subset, targets, actual_compositions = build_known_targets(data, descriptors)
    i1, i2, d1, d2 = nearest_two(targets, candidates)
    descriptor_span = float(np.linalg.norm(np.ptp(np.vstack(list(descriptors.values())), axis=0)))
    records = []
    for pos, (_, row) in enumerate(subset.iterrows()):
        actual = actual_compositions[pos]
        predicted = candidate_compositions[i1[pos]]
        second = candidate_compositions[i2[pos]]
        errors = composition_errors(actual, predicted)
        grid_compatible = bool(
            abs(sum(r for _, r in actual) - 1.0) <= 1e-10
            and all(abs(r / GRID_STEP - round(r / GRID_STEP)) <= 1e-10 for _, r in actual)
        )
        records.append(
            {
                "source_row": int(row["source_row"]),
                "k": float(row["k"]),
                "T": float(row["T"]),
                "c": float(row["c"]),
                "salt": str(row["salt"]),
                "solvent_count": len(actual),
                "actual_composition": format_composition(actual),
                "actual_ratio_sum": float(sum(r for _, r in actual)),
                "on_0_05_grid": grid_compatible,
                "target_descriptor_1": float(targets[pos, 0]),
                "target_descriptor_2": float(targets[pos, 1]),
                "predicted_composition": format_composition(predicted),
                "predicted_descriptor_1": float(candidates[i1[pos], 0]),
                "predicted_descriptor_2": float(candidates[i1[pos], 1]),
                "descriptor_distance": float(d1[pos]),
                "descriptor_distance_relative_to_library_span": float(d1[pos] / descriptor_span),
                "second_composition": format_composition(second),
                "second_descriptor_distance": float(d2[pos]),
                "nearest_neighbor_margin": float(d2[pos] - d1[pos]),
                **errors,
            }
        )
    return pd.DataFrame.from_records(records)


def sensitivity_analysis(
    detail: pd.DataFrame,
    descriptors: dict[str, np.ndarray],
    candidates: np.ndarray,
    candidate_compositions: list[tuple[tuple[str, float], ...]],
    seed: int = 20261001,
) -> pd.DataFrame:


    unique = detail.drop_duplicates("actual_composition").reset_index(drop=True)
    targets = unique[["target_descriptor_1", "target_descriptor_2"]].to_numpy(float)
    actual = []
    for value in unique["actual_composition"]:
        parts = []
        for item in value.split(" + "):
            name, ratio = item.rsplit(":", 1)
            parts.append((name, float(ratio)))
        actual.append(tuple(parts))

    values = np.vstack(list(descriptors.values()))
    feature_ranges = np.ptp(values, axis=0)
    rng = np.random.default_rng(seed)
    rows = []
    for fraction in (0.0, 0.001, 0.005, 0.01, 0.02):
        identity_hits = []
        tol_hits = []
        distances = []
        replicates = 20 if fraction > 0 else 1
        for _ in range(replicates):
            perturbed = targets + rng.normal(size=targets.shape) * feature_ranges * fraction
            i1, _, d1, _ = nearest_two(perturbed, candidates)
            for idx, candidate_idx in enumerate(i1):
                errors = composition_errors(actual[idx], candidate_compositions[candidate_idx])
                identity_hits.append(bool(errors["identity_match"]))
                tol_hits.append(bool(errors["composition_match_tol_0_05"]))
                distances.append(float(d1[idx]))
        rows.append(
            {
                "perturbation_fraction_of_feature_range": fraction,
                "unique_known_compositions": len(unique),
                "replicates": replicates,
                "identity_recovery_rate": float(np.mean(identity_hits)),
                "composition_recovery_rate_tol_0_05": float(np.mean(tol_hits)),
                "mean_nearest_descriptor_distance": float(np.mean(distances)),
                "median_nearest_descriptor_distance": float(np.median(distances)),
            }
        )
    return pd.DataFrame(rows)


def summarize(
    data: pd.DataFrame, solvent_detail: pd.DataFrame, salt_detail: pd.DataFrame
) -> pd.DataFrame:
    component_count = data[[f"solvent_{i}" for i in range(1, 5)]].notna().sum(axis=1)
    rows: list[dict[str, object]] = []

    def add(section: str, metric: str, value: object, population: str = "") -> None:
        rows.append({"section": section, "metric": metric, "value": value, "population": population})

    add("Dataset", "All formulations", len(data), "All rows")
    for count in (1, 2, 3, 4):
        n = int((component_count == count).sum())
        add("Dataset", f"{count}-component formulations", n, f"{n / len(data):.6%} of all rows")
    add(
        "Dataset",
        "Rows covered by unary/binary readout analysis",
        len(solvent_detail),
        f"{len(solvent_detail) / len(data):.6%} of all rows",
    )
    add("Salt", "Top-1 identity recovery rate", salt_detail["salt_identity_match"].mean(), "All formulations")
    add("Salt", "Mean nearest descriptor distance", salt_detail["salt_descriptor_distance"].mean(), "All formulations")

    for count, label in ((1, "Unary"), (2, "Binary"), (None, "Unary + binary")):
        part = solvent_detail if count is None else solvent_detail[solvent_detail["solvent_count"] == count]
        add(label, "Number of samples", len(part), label)
        add(label, "Fraction exactly on 0.05 grid", part["on_0_05_grid"].mean(), label)
        add(label, "Solvent identity recovery rate", part["identity_match"].mean(), label)
        add(label, "Strict identity + raw ratio recovery rate", part["strict_composition_match"].mean(), label)
        add(label, "Identity + normalized ratio recovery rate (tol 0.025)", part["composition_match_tol_0_025"].mean(), label)
        add(label, "Identity + normalized ratio recovery rate (tol 0.05)", part["composition_match_tol_0_05"].mean(), label)
        add(label, "Mean descriptor distance", part["descriptor_distance"].mean(), label)
        add(label, "Median descriptor distance", part["descriptor_distance"].median(), label)
        add(label, "95th percentile descriptor distance", part["descriptor_distance"].quantile(0.95), label)
        add(label, "Mean normalized ratio max-absolute error", part["ratio_max_abs_error_normalized"].mean(), label)
        add(label, "Median nearest-neighbor margin", part["nearest_neighbor_margin"].median(), label)
        add(label, "Fraction with effectively tied top two candidates", (part["nearest_neighbor_margin"] <= 1e-10).mean(), label)

    add("Comparator", "Explicit solvent composition dimension", 38, "One coordinate per solvent")
    add("Comparator", "Explicit representation exact recovery", 1.0, "By construction within the fixed library")
    add("Comparator", "Weighted HOMO-LUMO dimension", 2, "Continuous compact descriptor")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/training_data.csv"))
    parser.add_argument("--salt", type=Path, default=Path("data/salt_MO.txt"))
    parser.add_argument("--solvent", type=Path, default=Path("data/solvent_MO.txt"))
    parser.add_argument("--output", type=Path, default=Path("analysis/mo_readout"))
    args = parser.parse_args()

    data = pd.read_csv(args.data, na_values=["null"])
    salt_descriptors = load_descriptors(args.salt)
    solvent_descriptors = load_descriptors(args.solvent, aliases=SOLVENT_ALIASES)
    candidates, candidate_compositions = make_solvent_candidates(solvent_descriptors)

    missing_salts = sorted(set(data["salt"].dropna()) - set(salt_descriptors))
    solvent_columns = [f"solvent_{i}" for i in range(1, 5)]
    observed_solvents = set(pd.unique(data[solvent_columns].to_numpy().ravel()))
    observed_solvents = {x for x in observed_solvents if pd.notna(x)}
    missing_solvents = sorted(observed_solvents - set(solvent_descriptors))
    if missing_salts or missing_solvents:
        raise ValueError(f"Missing descriptors: salts={missing_salts}, solvents={missing_solvents}")

    salt_detail = salt_recovery(data, salt_descriptors)
    solvent_detail = solvent_recovery(
        data, solvent_descriptors, candidates, candidate_compositions
    )
    sensitivity = sensitivity_analysis(
        solvent_detail, solvent_descriptors, candidates, candidate_compositions
    )
    summary = summarize(data, solvent_detail, salt_detail)

    args.output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "mo_readout_summary.csv", index=False)
    solvent_detail.to_csv(args.output / "mo_readout_per_sample.csv", index=False)
    salt_detail.to_csv(args.output / "salt_readout_per_sample.csv", index=False)
    sensitivity.to_csv(args.output / "mo_readout_sensitivity.csv", index=False)
    metadata = {
        "data_file": str(args.data),
        "salt_descriptor_file": str(args.salt),
        "solvent_descriptor_file": str(args.solvent),
        "grid_step": GRID_STEP,
        "candidate_count": len(candidates),
        "pure_candidate_count": len(solvent_descriptors),
        "binary_candidate_count": len(candidates) - len(solvent_descriptors),
        "evaluated_sample_count": len(solvent_detail),
        "random_seed": 20261001,
        "sensitivity_replicates": 20,
        "solvent_aliases": SOLVENT_ALIASES,
    }
    (args.output / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\nSensitivity\n", sensitivity.to_string(index=False))
    print("\nMetadata\n", json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
