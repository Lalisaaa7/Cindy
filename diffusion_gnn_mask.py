# diffusion_gnn_mask.py
# A complete framework for predicting protein-DNA binding sites using a diffusion model + GNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, recall_score, f1_score
import random
import numpy as np


# --- 1. Diffusion Schedule ---
def get_diffusion_schedule(T, beta_1=1e-4, beta_T=0.02):
    betas = torch.linspace(beta_1, beta_T, T)
    alphas = 1. - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_bar


# --- 2. Time Embedding ---
class TimeEmbedding(nn.Module):
    def __init__(self, T, dim):
        super().__init__()
        self.embed = nn.Embedding(T, dim)
        self.linear = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, t):
        x = self.embed(t)
        return self.linear(x)


# --- 3. GNN + Diffusion ---
class DiffusionGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, T):
        super().__init__()
        self.time_embed = TimeEmbedding(T, hidden_dim)
        self.conv1 = pyg_nn.SAGEConv(in_dim + hidden_dim, hidden_dim)
        self.conv2 = pyg_nn.SAGEConv(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)  # predict noise (scalar per node)

    def forward(self, x, edge_index, t):
        t_embed = self.time_embed(t).unsqueeze(1).repeat(1, x.size(0), 1)
        x = torch.cat([x, t_embed[0]], dim=-1)
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return self.out(h).squeeze(-1)  # [N]


# --- 4. Training Step ---
def train_step(model, data, t, alphas_bar, device):
    model.train()
    x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.float().to(device)
    eps = torch.randn_like(y)
    sqrt_ab = torch.sqrt(alphas_bar[t]).to(device)
    sqrt_1m_ab = torch.sqrt(1 - alphas_bar[t]).to(device)
    y_t = sqrt_ab * y + sqrt_1m_ab * eps
    pred_eps = model(x, edge_index, torch.tensor([t]).to(device))
    loss = F.mse_loss(pred_eps, eps)
    return loss


# --- 5. Sampling (reverse process) ---
def sample(model, x, edge_index, alphas, alphas_bar, T, device):
    model.eval()
    y_t = torch.randn(x.size(0)).to(device)
    with torch.no_grad():
        for t in reversed(range(T)):
            t_tensor = torch.tensor([t]).to(device)
            beta = 1 - alphas[t]
            coeff1 = 1. / torch.sqrt(alphas[t])
            coeff2 = beta / torch.sqrt(1 - alphas_bar[t])
            eps = model(x.to(device), edge_index.to(device), t_tensor)
            mean = coeff1 * y_t - coeff2 * eps
            if t > 0:
                noise = torch.randn_like(y_t)
                y_t = mean + torch.sqrt(beta) * noise
            else:
                y_t = mean
    return torch.sigmoid(y_t)


# --- 6. Training Loop ---
def train_model(dataset, in_dim, hidden_dim, T, epochs, lr, device):
    betas, alphas, alphas_bar = get_diffusion_schedule(T)
    model = DiffusionGNN(in_dim, hidden_dim, T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for data in loader:
            t = random.randint(0, T - 1)
            optimizer.zero_grad()
            loss = train_step(model, data, t, alphas_bar, device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

    return model, betas, alphas, alphas_bar


# --- 7. Evaluation ---
def evaluate_model(model, dataset, alphas, alphas_bar, T, device, threshold=0.5):
    model.eval()
    real_label = []
    predict_label = []
    probabilities = []

    for data in dataset:
        x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
        pred_prob = sample(model, x, edge_index, alphas, alphas_bar, T, device)
        prob = pred_prob.detach().cpu().numpy()
        pred = (prob >= threshold).astype(int)
        probabilities.extend(prob.tolist())
        predict_label.extend(pred.tolist())
        real_label.extend(y.cpu().numpy().tolist())

    TN, FP, FN, TP = confusion_matrix(real_label, predict_label).ravel()
    spe = TN / (TN + FP)
    rec = recall_score(real_label, predict_label)
    pre = TP / (TP + FP)
    f1 = f1_score(real_label, predict_label)
    mcc = matthews_corrcoef(real_label, predict_label)
    auc = roc_auc_score(real_label, probabilities)

    print('Test Set Spe: {:.4f}'.format(spe))
    print('Test Set Rec: {:.4f}'.format(rec))
    print('Test Set Pre: {:.4f}'.format(pre))
    print('Test Set F1: {:.4f}'.format(f1))
    print('Test Set MCC: {:.4f}'.format(mcc))
    print('Test Set AUC: {:.4f}'.format(auc))

    return {
        'spe': spe, 'rec': rec, 'pre': pre,
        'f1': f1, 'mcc': mcc, 'auc': auc
    }