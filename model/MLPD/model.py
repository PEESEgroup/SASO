import numpy as np
import torch
import torch.nn as nn

def make_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)

class DiffusionNoiseScheduler:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = make_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

    def add_noise(self, y, noise, t):
        # y: [B, D], noise: [B, D], t: [B]
        alpha_bar = torch.tensor(self.alpha_bars[t.cpu()], dtype=torch.float32).to(y.device).unsqueeze(1)
        return (alpha_bar.sqrt() * y + (1 - alpha_bar).sqrt() * noise)


class MLPDiffusionModel(nn.Module):
    def __init__(self, input_dim, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + cond_dim + 1, 128),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, y_noisy, cond, t):
        # y_noisy, cond: [B, D]; t: [B, 1]
        x = torch.cat([y_noisy, cond, t], dim=1)
        return self.net(x)

