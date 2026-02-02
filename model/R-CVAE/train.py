import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from descriptor_utils import ElectrolyteDataset
from model_r import CVAEWithRouting as CVAE
import numpy as np
from smiles import salt_features_dict, solvent_features_dict
from torch.utils.data import Dataset, DataLoader

# --------- ✅ 样本权重计算函数 ---------
def compute_sample_weights(y, num_bins=5):
    y = y.detach().cpu().numpy().flatten()
    hist, bin_edges = np.histogram(y, bins=num_bins)
    bin_indices = np.digitize(y, bin_edges[:-1], right=True)
    freq = np.bincount(bin_indices, minlength=num_bins+1)[1:]
    freq = freq / np.sum(freq)
    raw_weights = 1.0 / (freq[bin_indices - 1] + 1e-6)
    alpha = 0.3
    weights = alpha * raw_weights + (1 - alpha) * 1.0
    weights = weights / np.mean(weights)  # normalize
    return torch.tensor(weights, dtype=torch.float32)

# --------- ✅ 修改 Dataset 类，支持权重 ---------
class WeightedElectrolyteDataset(Dataset):
    def __init__(self, df, solvent_feature_dict, salt_feature_dict):
        self.dataset = ElectrolyteDataset(df, solvent_feature_dict, salt_feature_dict)
        self.X, self.Y = self.dataset.get_features_and_targets()
        self.weights = compute_sample_weights(self.Y[:, 0])  # 假设Y[:, 0]是目标电导率
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)
        self.weights = torch.tensor(self.weights, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.weights[idx]

# --------- ✅ 加载数据 ---------
df = pd.read_csv("compressed_new.csv")
solvent_feature_dict = solvent_features_dict
salt_feature_dict = salt_features_dict
dataset = ElectrolyteDataset(df, solvent_feature_dict, salt_feature_dict)
X_raw, Y_raw = dataset.get_features_and_targets()
X, Y = torch.tensor(X_raw).float(), torch.tensor(Y_raw).float()
#dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# --------- ✅ 初始化模型 ---------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CVAE(cond_dim=X.shape[1], out_dim=Y.shape[1]).to(device)
opt = optim.Adam(model.parameters(), lr=1e-3)

# --------- ✅ 自定义 loss 函数，支持 weights ---------
def loss_fn(recon_x, x, mu, logvar, weights):
    recon_loss = F.mse_loss(recon_x, x, reduction='none').sum(dim=1)  # [batch_size]
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # [batch_size]
    loss = recon_loss * weights + kl_loss * weights
    return loss.mean()

# --------- ✅ 训练循环 ---------
for epoch in range(1000):
    model.train()
    total_loss = 0.0
    X, Y = X.to(device), Y.to(device)
    recon, mu, logvar = model(Y, X)
    weights = compute_sample_weights(Y[:, 0])  # 以目标值的第一个维度为主
    weights = weights.to(device)
    #loss = loss_fn(recon, Y, mu, logvar, weights)
    mse = nn.MSELoss()(recon, Y)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = mse + kl
    loss = (loss * weights).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"[{epoch}] Loss: {loss:.4f}")

# --------- ✅ 保存模型 ---------
torch.save(model.state_dict(), "cvae.pt")

