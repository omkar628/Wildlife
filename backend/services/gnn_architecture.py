"""Inference-only copy of the DistanceAwareGNN classes.

Copied from ``gnn_model_v3_optimized.py`` so production can load the trained
``.pt`` weights without importing that file's pandas / training stack.

Do not change layer sizes, forward signature, or masking behavior.
The training script and weights stay untouched.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.10):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_norm):
        x = torch.matmul(adj_norm, x)
        x = self.linear(x)
        x = F.relu(x)
        return self.dropout(x)


class CameraGCN(nn.Module):
    def __init__(self, in_dim, hidden=96, out_dim=96):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden)
        self.gcn2 = GCNLayer(hidden, out_dim)

    def forward(self, x, adj_norm):
        x = self.gcn1(x, adj_norm)
        x = self.gcn2(x, adj_norm)
        return x


class DistanceAwareGNN(nn.Module):
    def __init__(self, camera_dim, candidate_extra_dim=9,
                 hidden=96, gru_hidden=128):
        super().__init__()

        self.camera_gnn = CameraGCN(
            in_dim=camera_dim,
            hidden=hidden,
            out_dim=hidden,
        )

        self.history_gru = nn.GRU(
            input_size=hidden + 1,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        score_in = gru_hidden + hidden + candidate_extra_dim

        self.scorer = nn.Sequential(
            nn.Linear(score_in, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        camera_x,
        adj_norm,
        history_idx,
        history_dt,
        candidate_idx,
        candidate_extra,
        candidate_mask,
    ):
        camera_emb = self.camera_gnn(camera_x, adj_norm)

        hist_emb = camera_emb[history_idx]
        hist_in = torch.cat([hist_emb, history_dt], dim=-1)

        _, h = self.history_gru(hist_in)
        tiger_state = h[-1]

        cand_emb = camera_emb[candidate_idx]
        B, C, _ = cand_emb.shape

        tiger_expand = tiger_state.unsqueeze(1).expand(-1, C, -1)

        score_input = torch.cat(
            [tiger_expand, cand_emb, candidate_extra],
            dim=-1,
        )

        logits = self.scorer(score_input).squeeze(-1)

        logits = logits.masked_fill(
            ~candidate_mask,
            torch.finfo(logits.dtype).min,
        )

        return logits
