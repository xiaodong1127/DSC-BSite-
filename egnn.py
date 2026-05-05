# -*- coding: utf-8 -*-
# eg nn.py — EGNN backbone without subgraph/pooling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

# ---------- EGCL ----------
class EGCL(nn.Module):
    def __init__(self, in_dim, hidden_dim, edge_dim=0):
        super().__init__()
        self.phi_e = nn.Sequential(
            nn.Linear(in_dim * 2 + 1 + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.phi_x = nn.Linear(hidden_dim, 1)
        self.phi_h = nn.Sequential(
            nn.Linear(in_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, h, pos, edge_index, edge_attr=None):
        if edge_index is None or edge_index.numel() == 0:
            return h, pos

        row, col = edge_index
        rel = pos[row] - pos[col]
        d2 = (rel ** 2).sum(dim=-1, keepdim=True)
        if edge_attr is None:
            m_ij = torch.cat([h[row], h[col], d2], dim=-1)
        else:
            m_ij = torch.cat([h[row], h[col], d2, edge_attr], dim=-1)
        m_ij = self.phi_e(m_ij)

        w_ij = self.phi_x(m_ij)
        coord_msg = rel * w_ij
        C = 1.0 / max(1, int(h.size(0)) - 1)
        delta_x = C * scatter_add(coord_msg, row, dim=0, dim_size=h.size(0))
        pos_new = pos + delta_x

        m_i = scatter_add(m_ij, row, dim=0, dim_size=h.size(0))
        h_new = self.phi_h(torch.cat([h, m_i], dim=-1))
        return h_new, pos_new

# ---------- ContinueModel ----------
class ContinueModel(nn.Module):
    def __init__(self, in_dim=1280, dropout=0.3):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, 1024), nn.LayerNorm(1024), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.layer5 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Softmax(-1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.layer1(x)
        inter = x
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(self.layer5(x)), inter

# ---------- EGNN Backbone without pooling ----------
class EGNNBackbone(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_layers=3, project_to=1280, k=16):
        super().__init__()
        self.enc_in = nn.Linear(in_dim, hidden_dim)
        self.k = k
        self.egcls_full = nn.ModuleList([EGCL(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.proj = nn.Linear(hidden_dim, project_to)
        self.head = ContinueModel(in_dim=project_to)

    def forward(self, x, pos, mask=None):
        """
        x: [N, F] 节点特征
        pos: [N, 3] 坐标
        """
        N = x.size(0)
        device = x.device
        h = F.silu(self.enc_in(x))

        # --- 构建 KNN 边 ---
        with torch.no_grad():
            if N == 1:
                edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
            else:
                k_eff = min(self.k + 1, N)
                dist = torch.cdist(pos, pos)
                _, knn_idx = torch.topk(dist, k=k_eff, largest=False)
                row, col = [], []
                for i in range(N):
                    neighbors = knn_idx[i, 1:].tolist()  # 排除自己
                    row.extend([i]*len(neighbors))
                    col.extend(neighbors)
                row = torch.tensor(row, dtype=torch.long, device=device)
                col = torch.tensor(col, dtype=torch.long, device=device)
                edge_index = torch.stack([row, col], dim=0)

        # --- 全图 EGNN ---
        for layer in self.egcls_full:
            h_res, pos = layer(h, pos, edge_index)
            h = F.relu(h + h_res)

        # --- 投影 & MLP ---
        z = self.proj(h)
        score, embed = self.head(z)
        return score, embed
