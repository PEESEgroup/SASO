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

    def encode(self, x, c):
        h1 = F.relu(self.fc1(torch.cat([x, c], dim=1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, c):
        h3 = F.relu(self.fc3(torch.cat([z, c], dim=1)))
        return self.fc4(h3)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
