import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from descriptor_utils import ElectrolyteDataset, load_descriptor_libraries
from model_r import CVAEWithRouting
from training_utils import cvae_loss, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "training_data.csv")
    parser.add_argument("--loss", choices=("unweighted", "weighted"), default="unweighted")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--experts", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-bins", type=int, default=5)
    parser.add_argument("--weight-alpha", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    salt_descriptors, solvent_descriptors = load_descriptor_libraries(ROOT / "data")
    data = pd.read_csv(args.data, na_values=["null"])
    dataset = ElectrolyteDataset(data, solvent_descriptors, salt_descriptors)
    formulations_raw, Y_raw = dataset.get_features_and_targets()
    formulations = torch.as_tensor(formulations_raw, dtype=torch.float32, device=args.device)
    Y = torch.as_tensor(Y_raw, dtype=torch.float32, device=args.device)
    model = CVAEWithRouting(
        cond_dim=Y.shape[1],
        out_dim=formulations.shape[1],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_experts=args.experts,
    ).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    weighted = args.loss == "weighted"
    for epoch in range(args.epochs):
        reconstruction, mean, log_variance = model(formulations, Y)
        loss = cvae_loss(
            reconstruction,
            formulations,
            mean,
            log_variance,
            Y,
            weighted,
            args.weight_bins,
            args.weight_alpha,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch < 10 or epoch % 10 == 0:
            print(f"[{epoch}] Loss: {loss.item():.6f}")
    output = args.output or ROOT / "checkpoints" / f"R-CVAE_{args.loss}.pt"
    save_checkpoint(
        model,
        output,
        {
            "architecture": "R-CVAE",
            "loss": args.loss,
            "condition": "Y[:,0] = ionic conductivity k",
            "data": str(args.data),
            "epochs": args.epochs,
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "experts": args.experts,
            "learning_rate": args.learning_rate,
            "weight_bins": args.weight_bins,
            "weight_alpha": args.weight_alpha if weighted else None,
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
