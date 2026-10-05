import numpy as np
import torch
import torch.nn as nn


def make_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)


class DiffusionNoiseScheduler:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = make_beta_schedule(timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

    def add_noise(self, formulation, noise, timestep):
        alpha_bar = torch.as_tensor(
            self.alpha_bars[timestep.detach().cpu().numpy()],
            dtype=formulation.dtype,
            device=formulation.device,
        ).unsqueeze(1)
        return alpha_bar.sqrt() * formulation + (1.0 - alpha_bar).sqrt() * noise


class MLPDiffusionModelWithRouting(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim + cond_dim + 1, hidden_dim), nn.ReLU()
        )
        self.route_a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.route_b = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.gate = nn.Sequential(nn.Linear(cond_dim + 1, 1), nn.Sigmoid())

    def forward(self, noisy_formulation, conductivity, timestep_embedding):
        inputs = torch.cat(
            [noisy_formulation, conductivity, timestep_embedding], dim=1
        )
        hidden = self.shared(inputs)
        gate_inputs = torch.cat([conductivity, timestep_embedding], dim=1)
        gate_weight = self.gate(gate_inputs)
        return gate_weight * self.route_b(hidden) + (1.0 - gate_weight) * self.route_a(hidden)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        mapped = {}
        for key, value in state_dict.items():
            key = key.replace("route_main", "route_a").replace("route_tail", "route_b")
            mapped[key] = value
        return super().load_state_dict(mapped, strict=strict, assign=assign)
