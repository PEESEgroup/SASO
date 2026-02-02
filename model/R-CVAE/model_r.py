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

        # Encoder
        self.fc1 = nn.Linear(cond_dim + out_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

        # Routing / gating network
        self.gate = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_experts),
            nn.Softmax(dim=1)
        )

        # Expert decoders (n_experts)
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + cond_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
            for _ in range(n_experts)
        ])

    def encode(self, x, c):
        h1 = F.relu(self.fc1(torch.cat([x, c], dim=1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, c):
        """
        Shared decode logic for both training and generation.
        z: [B, latent_dim], c: [B, cond_dim]
        """
        gate_input = torch.cat([z, c], dim=1)
        gate_weights = self.gate(gate_input)  # [B, n_experts]

        all_outputs = []
        for decoder in self.decoders:
            out_i = decoder(torch.cat([z, c], dim=1))  # [B, out_dim]
            all_outputs.append(out_i.unsqueeze(2))     # [B, out_dim, 1]

        all_outputs = torch.cat(all_outputs, dim=2)     # [B, out_dim, n_experts]
        gate_weights = gate_weights.unsqueeze(1)        # [B, 1, n_experts]

        output = torch.bmm(all_outputs, gate_weights.transpose(1, 2)).squeeze(2)  # [B, out_dim]
        return output

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

