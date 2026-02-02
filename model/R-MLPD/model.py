# model.py

import torch
import torch.nn as nn
import numpy as np

def make_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)

class DiffusionNoiseScheduler:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = make_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

    def add_noise(self, y, noise, t):
        alpha_bar = torch.tensor(self.alpha_bars[t.cpu()], dtype=torch.float32).to(y.device).unsqueeze(1)
        return (alpha_bar.sqrt() * y + (1 - alpha_bar).sqrt() * noise)

# ✅ 动态路由 MLP
class MLPDiffusionModelWithRouting(nn.Module):
    def __init__(self, input_dim, cond_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim + cond_dim + 1, 128),
            nn.ReLU()
        )
        self.route_main = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        self.route_tail = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(cond_dim + 1, 1),  # cond + timestep
            nn.Sigmoid()  # 输出 [0,1]，越接近1越偏向 tail 路径
        )

    def forward(self, y_noisy, cond, t_embed):
        x = torch.cat([y_noisy, cond, t_embed], dim=1)
        h = self.shared(x)

        gate_input = torch.cat([cond, t_embed], dim=1)
        g = self.gate(gate_input)  # [B, 1]

        out_main = self.route_main(h)
        out_tail = self.route_tail(h)
        return g * out_tail + (1 - g) * out_main

