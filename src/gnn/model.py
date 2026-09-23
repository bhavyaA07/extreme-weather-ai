"""
Spatio-temporal GNN that scores each candidate edge (event_t -> event_t+1)
with the probability that both nodes belong to the same evolving weather
system. Track construction (tracker.py) then greedily/optimally links the
highest-scoring edges into full trajectories.

Architecture: a couple of GraphSAGE/ EdgeConv-style message passing layers
over node features, followed by an MLP edge classifier that takes the two
endpoint embeddings + edge_attr (distance, EFI similarity).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class EventEncoder(nn.Module):
    def __init__(self, in_dim: int = 6, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

    def forward(self, x, edge_index):
        h = F.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)
            h = norm(h + F.relu(h_new))  # residual
        return h


class EdgeScorer(nn.Module):
    """Given two node embeddings + raw edge features, predicts P(same track)."""

    def __init__(self, hidden_dim: int = 64, edge_feat_dim: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h, edge_index, edge_attr):
        src, dst = edge_index
        pair = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        return self.mlp(pair).squeeze(-1)  # raw logits


class StormTrackGNN(nn.Module):
    """
    Full model: encode nodes -> score edges -> (optionally) predict the next
    centroid position for the storm-motion forecast head.
    """

    def __init__(self, in_dim: int = 6, hidden_dim: int = 64, n_layers: int = 3, edge_feat_dim: int = 2):
        super().__init__()
        self.encoder = EventEncoder(in_dim, hidden_dim, n_layers)
        self.edge_scorer = EdgeScorer(hidden_dim, edge_feat_dim)
        # motion head: predicts (delta_lat, delta_lon) to the next timestep centroid
        self.motion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x, edge_index, edge_attr):
        h = self.encoder(x, edge_index)
        edge_logits = self.edge_scorer(h, edge_index, edge_attr) if edge_index.numel() > 0 else torch.zeros(0)
        motion_pred = self.motion_head(h)
        return edge_logits, motion_pred, h


def track_loss(edge_logits, edge_labels, motion_pred, motion_target, motion_mask, motion_weight: float = 0.5):
    """
    edge_labels: 1 if the two events are truly the same storm (from labeled
                 training tracks, e.g. IBTrACS best-track association), else 0.
    motion_target/motion_mask: ground-truth (delta_lat, delta_lon) where available.
    """
    edge_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels.float())
    if motion_mask.sum() > 0:
        motion_loss = F.mse_loss(motion_pred[motion_mask], motion_target[motion_mask])
    else:
        motion_loss = torch.tensor(0.0, device=edge_logits.device)
    return edge_loss + motion_weight * motion_loss, edge_loss.item(), motion_loss.item()
