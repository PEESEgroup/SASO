import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAEWithRouting(nn.Module):
    def __init__(self, cond_dim, out_dim, latent_dim=16, hidden_dim=64, n_experts=5):
        super().__init__()
        self.cond_dim = cond_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.n_experts = n_experts
        self.fc1 = nn.Linear(cond_dim + out_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.gate = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts),
            nn.Softmax(dim=1),
        )
        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(latent_dim + cond_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_dim),
                )
                for _ in range(n_experts)
            ]
        )

    def encode(self, formulation, conductivity):
        hidden = F.relu(self.fc1(torch.cat([formulation, conductivity], dim=1)))
        return self.fc21(hidden), self.fc22(hidden)

    def reparameterize(self, mean, log_variance):
        standard_deviation = torch.exp(0.5 * log_variance)
        return mean + standard_deviation * torch.randn_like(standard_deviation)

    def decode(self, latent, conductivity):
        inputs = torch.cat([latent, conductivity], dim=1)
        gate_weights = self.gate(inputs)
        expert_outputs = torch.stack(
            [decoder(inputs) for decoder in self.decoders], dim=2
        )
        return torch.bmm(expert_outputs, gate_weights.unsqueeze(2)).squeeze(2)

    def forward(self, formulation, conductivity):
        mean, log_variance = self.encode(formulation, conductivity)
        latent = self.reparameterize(mean, log_variance)
        return self.decode(latent, conductivity), mean, log_variance
