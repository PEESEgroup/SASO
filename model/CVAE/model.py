import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    def __init__(self, cond_dim, out_dim, latent_dim=16, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim + out_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim + cond_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, out_dim)

    def encode(self, formulation, conductivity):
        hidden = F.relu(self.fc1(torch.cat([formulation, conductivity], dim=1)))
        return self.fc21(hidden), self.fc22(hidden)

    def reparameterize(self, mean, log_variance):
        standard_deviation = torch.exp(0.5 * log_variance)
        return mean + standard_deviation * torch.randn_like(standard_deviation)

    def decode(self, latent, conductivity):
        hidden = F.relu(self.fc3(torch.cat([latent, conductivity], dim=1)))
        return self.fc4(hidden)

    def forward(self, formulation, conductivity):
        mean, log_variance = self.encode(formulation, conductivity)
        latent = self.reparameterize(mean, log_variance)
        return self.decode(latent, conductivity), mean, log_variance
