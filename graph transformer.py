import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import subgraph
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import dense_to_sparse


# ---------- Pooling & Unpooling ----------
class SAGPool(nn.Module):
    def __init__(self, feat_dim, ratio=0.5, min_nodes=20):
        super().__init__()
        self.ratio = ratio
        self.min_nodes = min_nodes
        self.score_layer = nn.Linear(feat_dim, 1)  # 可训练打分

    def _topk(self, score, k):
        return torch.topk(score, k=k, largest=True, sorted=True).indices

    def forward(self, X, edge_index):
        score = self.score_layer(X).squeeze()
        score_sig = torch.sigmoid(score)

        N = X.size(0)
        k = max(self.min_nodes, int(self.ratio * N))
        idx = self._topk(score_sig, k)

        mask = torch.zeros(N, dtype=torch.bool, device=X.device)
        mask[idx] = True
        edge_index_sub, _ = subgraph(mask, edge_index, relabel_nodes=True)

        X_sub = X[idx]
        return X_sub, edge_index_sub, idx, score_sig


class GraphUnpool(nn.Module):
    def forward(self, X_sub, idx, N):
        X_full = X_sub.new_zeros(N, X_sub.size(-1))
        X_full[idx] = X_sub
        return X_full


# ---------- ContinueModel ----------
class ContinueModel(nn.Module):
    def __init__(self, in_dim=1280, dropout=0.3):
        super(ContinueModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.layer5 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Softmax(-1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.layer1(x)
        inter = x  # dim 1024

        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(self.layer5(x)), inter


# ---------- Fusion Module ----------
class FusionModule(nn.Module):
    def __init__(self, hidden_dim, fuse_dim=256):
        super().__init__()
        self.fuse_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, fuse_dim),
            nn.ReLU(),
            nn.Linear(fuse_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.gate_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h_full, h_sub_up):
        h_full = self.norm1(h_full)
        h_sub_up = self.norm2(h_sub_up)

        h_cat = torch.cat([h_full, h_sub_up], dim=-1)
        h_agg = self.fuse_mlp(h_cat)

        h_out = h_full + h_agg

        gate = torch.sigmoid(self.gate_linear(h_sub_up))
        h_out = h_out + gate * h_sub_up

        return h_out


# ---------- Graph Transformer Backbone ----------
class GraphTransformerWithSAGPoolBackbone(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_layers=3, pool_ratio=0.5, project_to=1280, heads=4):
        super().__init__()
        self.enc_in = nn.Linear(in_dim, hidden_dim)

        # Graph Transformer layers
        self.trans_full = nn.ModuleList(
            [TransformerConv(hidden_dim, hidden_dim, heads=heads, edge_dim=None) for _ in range(num_layers)]
        )
        self.trans_sub = nn.ModuleList(
            [TransformerConv(hidden_dim, hidden_dim, heads=heads, edge_dim=None) for _ in range(num_layers)]
        )

        self.pool = SAGPool(feat_dim=hidden_dim, ratio=pool_ratio, min_nodes=20)
        self.unpool = GraphUnpool()
        self.fusion = FusionModule(hidden_dim, fuse_dim=hidden_dim)

        self.proj = nn.Linear(hidden_dim, project_to)
        self.head = ContinueModel(in_dim=project_to)

    def forward(self, x, edge_index=None):
        """
        x: [N, F] node features
        edge_index: [2, E] edge indices (可选，如果不给就自动生成全连接图)
        """
        N = x.size(0)
        h0 = F.silu(self.enc_in(x))

        # ---- 如果没有传入 edge_index，就生成全连接图 ----
        if edge_index is None:
            adj = torch.ones((N, N), device=x.device) - torch.eye(N, device=x.device)  # 去掉自环
            edge_index, _ = dense_to_sparse(adj)

        # --- 子图 ---
        X_sub, edge_index_sub, idx, _ = self.pool(h0, edge_index)

        # --- 全图 Graph Transformer ---
        h_full = h0
        for layer in self.trans_full:
            h_full = F.relu(layer(h_full, edge_index))

        # --- 子图 Graph Transformer ---
        h_sub = X_sub
        for layer in self.trans_sub:
            h_sub = F.relu(layer(h_sub, edge_index_sub))

        # --- unpool 回原图 ---
        h_sub_up = self.unpool(h_sub, idx, N=N)

        # --- 融合 ---
        h_agg = self.fusion(h_full, h_sub_up)

        # --- 投影 + 分类 ---
        z = self.proj(h_agg)
        score, embed = self.head(z)
        return score, embed
