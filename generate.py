import argparse
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model" / "R-CVAE"))

from descriptor_utils import ElectrolyteDataset, load_descriptor_libraries
from model_r import CVAEWithRouting
from search_salt_solvent import search_best_2solvent, search_best_salt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "trained_pt" / "R-CVAE.pt")
    parser.add_argument("--target-k", type=float, default=20.0)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "generated_20.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    salts, solvents = load_descriptor_libraries(ROOT / "data")
    data = pd.read_csv(ROOT / "data" / "training_data.csv", na_values=["null"])
    dataset = ElectrolyteDataset(data, solvents, salts)
    formulations, Y = dataset.get_features_and_targets()
    model = CVAEWithRouting(cond_dim=Y.shape[1], out_dim=formulations.shape[1])
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    results = []
    for _ in range(args.samples):
        Y_target = torch.tensor([[args.target_k]], dtype=torch.float32)
        latent = torch.randn(1, model.latent_dim)
        with torch.no_grad():
            generated = model.decode(latent, Y_target).numpy()
        temperature_concentration, categories, salt_vectors, solvent_vectors = dataset.decode(generated)
        salt, _ = search_best_salt(salt_vectors[0], salts)
        solvent, _ = search_best_2solvent(solvent_vectors[0], solvents)
        results.append(
            {
                "T": temperature_concentration[0, 0],
                "c": temperature_concentration[0, 1],
                "c units": categories[0, 0],
                "solvent ratio type": categories[0, 1],
                "salt": salt,
                "solvent_1": solvent[0],
                "ratio_1": solvent[1],
                "solvent_2": solvent[2],
                "ratio_2": solvent[3],
                "k": args.target_k,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
