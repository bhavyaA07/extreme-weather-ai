"""
Runs the trained StormTrackGNN on a fresh forecast's detected events and
produces final storm tracks: chains of events across lead times, each
representing one evolving extreme-weather system.

Approach: score every candidate edge, then greedily build tracks by always
extending from the highest-scoring outgoing edge above `edge_threshold`
(simple, fast, good enough for a hackathon demo — swap for a proper
min-cost-flow / Hungarian assignment later if time allows).
"""

from __future__ import annotations
from typing import List, Dict
import torch

from src.gnn.model import StormTrackGNN
from src.gnn.graph_builder import build_track_graph
from src.anomaly.detector import WeatherEvent


class StormTracker:
    def __init__(self, model_path: str, hidden_dim: int = 64, n_layers: int = 3, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = StormTrackGNN(in_dim=6, hidden_dim=hidden_dim, n_layers=n_layers).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def track(
        self,
        events_per_timestep: List[List[WeatherEvent]],
        edge_threshold: float = 0.5,
        max_link_distance_km: float = 300.0,
    ) -> List[List[str]]:
        """
        Returns a list of tracks, each a list of event_ids in temporal order,
        e.g. [["precipitation_t0_1", "precipitation_t1_2", "precipitation_t2_1"], ...]
        """
        data = build_track_graph(events_per_timestep, max_link_distance_km)
        if data.edge_index.numel() == 0:
            # no candidate links at all -> every event is its own singleton track
            return [[eid] for eid, _ in data.node_meta]

        data = data.to(self.device)
        edge_logits, motion_pred, _ = self.model(data.x, data.edge_index, data.edge_attr)
        edge_probs = torch.sigmoid(edge_logits).cpu().numpy()

        # adjacency: node -> best next-node above threshold
        best_next: Dict[int, tuple] = {}
        for e in range(data.edge_index.shape[1]):
            src = data.edge_index[0, e].item()
            dst = data.edge_index[1, e].item()
            prob = edge_probs[e]
            if prob < edge_threshold:
                continue
            if src not in best_next or prob > best_next[src][1]:
                best_next[src] = (dst, prob)

        node_meta = data.node_meta  # (event_id, lead_time_idx)
        has_incoming = {dst for dst, _ in best_next.values()}

        tracks: List[List[str]] = []
        visited = set()
        for start in range(len(node_meta)):
            if start in has_incoming or start in visited:
                continue
            chain = [start]
            cur = start
            visited.add(cur)
            while cur in best_next:
                nxt, _ = best_next[cur]
                if nxt in visited:
                    break
                chain.append(nxt)
                visited.add(nxt)
                cur = nxt
            tracks.append([node_meta[i][0] for i in chain])

        return tracks
