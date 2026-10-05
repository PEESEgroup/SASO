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
from model import DiffusionNoiseScheduler, MLPDiffusionModelWithRouting
from training_utils import reduce_per_sample_loss, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "training_data.csv")
    parser.add_argument("--loss", choices=("unweighted", "weighted"), default="unweighted")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-bins", type=int, default=5)
    parser.add_argument("--weight-alpha", type=float, default=0.5)
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
    model = MLPDiffusionModelWithRouting(
        input_dim=formulations.shape[1],
        cond_dim=Y.shape[1],
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    scheduler = DiffusionNoiseScheduler(timesteps=args.timesteps)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    weighted = args.loss == "weighted"
    for epoch in range(args.epochs):
        timestep = torch.randint(
            0, args.timesteps, (formulations.shape[0],), device=args.device
        )
        timestep_embedding = timestep.float().unsqueeze(1) / args.timesteps
        noise = torch.randn_like(formulations)
        noisy_formulations = scheduler.add_noise(formulations, noise, timestep)
        predicted_noise = model(noisy_formulations, Y, timestep_embedding)
        per_sample_loss = (predicted_noise - noise).pow(2).mean(dim=1)
        loss = reduce_per_sample_loss(
            per_sample_loss,
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
    output = args.output or ROOT / "checkpoints" / f"R-MLPD_{args.loss}.pth"
    save_checkpoint(
        model,
        output,
        {
            "architecture": "R-MLPD",
            "loss": args.loss,
            "condition": "Y[:,0] = ionic conductivity k",
            "data": str(args.data),
            "epochs": args.epochs,
            "timesteps": args.timesteps,
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "weight_bins": args.weight_bins,
            "weight_alpha": args.weight_alpha if weighted else None,
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
