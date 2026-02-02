# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from model import MLPDiffusionModelWithRouting, DiffusionNoiseScheduler
from smiles import salt_features_dict, solvent_features_dict
from descriptor_utils import ElectrolyteDataset

# 参数
timesteps = 500
epochs = 1000

# ✅ 权重计算函数
def compute_sample_weights(y, num_bins=5):
    y = y.detach().cpu().numpy().flatten()
    hist, bin_edges = np.histogram(y, bins=num_bins)
    bin_idx = np.digitize(y, bin_edges[:-1], right=True)
    freq = np.bincount(bin_idx, minlength=num_bins+1)[1:]
    freq = freq / np.sum(freq)
    raw_weights = 1. / (freq[bin_idx - 1] + 1e-6)
    alpha = 0.5
    weights = alpha * raw_weights + (1 - alpha) * 1.0
    weights = weights / np.mean(weights)  # normalize
    #weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)

# 加载数据
df = pd.read_csv("compressed_new.csv")
dataset = ElectrolyteDataset(df, solvent_features_dict, salt_features_dict)
X_raw, Y_raw = dataset.get_features_and_targets()
X, Y = torch.tensor(X_raw).float(), torch.tensor(Y_raw).float()

# 初始化模型与调度器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPDiffusionModelWithRouting(input_dim=Y.shape[1], cond_dim=X.shape[1]).to(device)
scheduler = DiffusionNoiseScheduler(timesteps=timesteps)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

X, Y = X.to(device), Y.to(device)

# 训练
for epoch in range(epochs):
    model.train()
    t = torch.randint(0, timesteps, (X.shape[0],)).long().to(device)
    t_embed = t.float().unsqueeze(1) / timesteps  # [B, 1]
    noise = torch.randn_like(Y)
    y_noisy = scheduler.add_noise(Y, noise, t)

    pred_noise = model(y_noisy, X, t_embed)

    # ✅ 加权 loss：按目标值的分布反比加权
    weights = compute_sample_weights(Y[:, 0])  # 以目标值的第一个维度为主
    weights = weights.to(device)

    loss = ((pred_noise - noise) ** 2).mean(dim=1)  # [B]
    loss = (loss * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch < 10:
        print(f"[{epoch}] Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "1-ddpm.pth")
