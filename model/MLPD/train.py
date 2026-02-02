import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from descriptor_utils import ElectrolyteDataset
import numpy as np
from model import MLPDiffusionModel, DiffusionNoiseScheduler
from smiles import salt_features_dict, solvent_features_dict


# 参数
timesteps = 500
epochs = 1000

# 加载数据
df = pd.read_csv("compressed_new.csv")
dataset = ElectrolyteDataset(df, solvent_features_dict, salt_features_dict)
X_raw, Y_raw = dataset.get_features_and_targets()
X, Y = torch.tensor(X_raw).float(), torch.tensor(Y_raw).float()

# 初始化模型与调度器
model = MLPDiffusionModel(input_dim=Y.shape[1], cond_dim=X.shape[1])
scheduler = DiffusionNoiseScheduler(timesteps=timesteps)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练
for epoch in range(epochs):
    t = torch.randint(0, timesteps, (X.shape[0],)).long()
    t_embed = t.float().unsqueeze(1) / timesteps
    noise = torch.randn_like(Y)
    y_noisy = scheduler.add_noise(Y, noise, t)

    pred_noise = model(y_noisy, X, t_embed)
    loss = nn.MSELoss()(pred_noise, noise)

    optimizer.zero_grad(); loss.backward(); optimizer.step()

    if epoch % 1 == 0:
        print(f"[{epoch}] Loss: {loss.item():.4f}")

# 保存模型
torch.save(model.state_dict(), "5-ddpm.pth")

