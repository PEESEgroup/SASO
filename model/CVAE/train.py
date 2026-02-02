import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from descriptor_utils import ElectrolyteDataset
from model_r import CVAEWithRouting as CVAE
import numpy as np
from smiles import salt_features_dict, solvent_features_dict

df = pd.read_csv("compressed_new.csv")
solvent_feature_dict = solvent_features_dict  # 2D 向量
salt_feature_dict = salt_features_dict

dataset = ElectrolyteDataset(df, solvent_feature_dict, salt_feature_dict)
X, Y = dataset.get_features_and_targets()
X, Y = torch.tensor(X).float(), torch.tensor(Y).float()

model = CVAE(cond_dim=X.shape[1], out_dim=Y.shape[1])
opt = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1000):
    recon, mu, logvar = model(Y, X)
    mse = nn.MSELoss()(recon, Y)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = mse + kl
    opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 1 == 0:
        print(f"[{epoch}] Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "r-cvae.pt")

