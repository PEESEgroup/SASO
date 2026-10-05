import random
import json
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_sample_weights(conductivity, num_bins=5, alpha=0.3):
    values = conductivity.detach().cpu().numpy().reshape(-1)
    edges = np.histogram_bin_edges(values, bins=num_bins)
    indices = np.digitize(values, edges[1:-1], right=False)
    counts = np.bincount(indices, minlength=num_bins).astype(float)
    frequencies = counts / counts.sum()
    inverse_frequency = 1.0 / (frequencies[indices] + 1e-6)
    weights = alpha * inverse_frequency + (1.0 - alpha)
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=conductivity.device)


def reduce_per_sample_loss(per_sample_loss, conductivity, weighted, num_bins, alpha):
    if not weighted:
        return per_sample_loss.mean()
    weights = compute_sample_weights(conductivity[:, 0], num_bins, alpha)
    return (per_sample_loss * weights).mean()


def cvae_loss(reconstruction, target, mean, log_variance, conductivity, weighted, num_bins, alpha):
    reconstruction_loss = (reconstruction - target).pow(2).mean(dim=1)
    reconstruction_loss = reduce_per_sample_loss(
        reconstruction_loss, conductivity, weighted, num_bins, alpha
    )
    kl_loss = -0.5 * (
        1 + log_variance - mean.pow(2) - log_variance.exp()
    ).mean(dim=1).mean()
    return reconstruction_loss + kl_loss


def save_checkpoint(model, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    record = dict(metadata)
    record["checkpoint"] = path.name
    record["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
