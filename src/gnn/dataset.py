"""
Dataset of (graph, edge_labels, motion_targets) built from historical
forecast runs where the "ground truth" track is known — e.g. by matching
detected events against IMD best-track / IBTrACS cyclone tracks, or against
ERA5 reanalysis-derived event tracks for non-cyclonic extremes (heavy rain
cells, heatwave blobs).

Label generation strategy:
  - For each pair of events in a candidate edge, label = 1 if both events'
    footprints intersect the same reference track's buffer at their
    respective valid times, else 0.
"""

from __future__ import annotations
from typing import List, Callable
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.gnn.graph_builder import build_track_graph, _haversine_km
from src.anomaly.detector import WeatherEvent


class StormTrackDataset(Dataset):
    def __init__(
        self,
        forecast_runs: List[List[List[WeatherEvent]]],
        reference_tracks: List[List[dict]],
        max_link_distance_km: float = 300.0,
        label_radius_km: float = 150.0,
    ):
        """
        forecast_runs[k]      : events_per_timestep for run k (see graph_builder)
        reference_tracks[k]   : list of {"lead_time_idx": int, "lat": float, "lon": float,
                                          "track_id": str} for run k's ground-truth track(s)
        """
        self.forecast_runs = forecast_runs
        self.reference_tracks = reference_tracks
        self.max_link_distance_km = max_link_distance_km
        self.label_radius_km = label_radius_km

    def __len__(self):
        return len(self.forecast_runs)

    def _label_edges(self, data: Data, ref_track: List[dict]) -> torch.Tensor:
        # Map each node to the nearest reference-track point at the same lead time;
        # a node is "on-track" if within label_radius_km of that point.
        node_track_id = [None] * len(data.node_meta)
        for i, (event_id, lead_time_idx) in enumerate(data.node_meta):
            candidates = [p for p in ref_track if p["lead_time_idx"] == lead_time_idx]
            # node lat/lon recovered from features (denormalised)
            lat = float(data.x[i, 0]) * 90.0
            lon = float(data.x[i, 1]) * 180.0
            for p in candidates:
                if _haversine_km(lat, lon, p["lat"], p["lon"]) <= self.label_radius_km:
                    node_track_id[i] = p["track_id"]
                    break

        edge_labels = []
        for e in range(data.edge_index.shape[1]):
            a, b = data.edge_index[0, e].item(), data.edge_index[1, e].item()
            same = (
                node_track_id[a] is not None
                and node_track_id[b] is not None
                and node_track_id[a] == node_track_id[b]
            )
            edge_labels.append(1.0 if same else 0.0)
        return torch.tensor(edge_labels, dtype=torch.float)

    def __getitem__(self, idx: int):
        events_per_timestep = self.forecast_runs[idx]
        ref_track = self.reference_tracks[idx]
        data = build_track_graph(events_per_timestep, self.max_link_distance_km)
        data.edge_labels = self._label_edges(data, ref_track)

        # motion targets: for on-track nodes, delta to the *next* reference point on the same track
        motion_target = torch.zeros((data.x.shape[0], 2))
        motion_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
        by_track = {}
        for p in ref_track:
            by_track.setdefault(p["track_id"], []).append(p)
        for tid, pts in by_track.items():
            pts_sorted = sorted(pts, key=lambda p: p["lead_time_idx"])
            for k in range(len(pts_sorted) - 1):
                cur, nxt = pts_sorted[k], pts_sorted[k + 1]
                for i, (event_id, lead_time_idx) in enumerate(data.node_meta):
                    if lead_time_idx != cur["lead_time_idx"]:
                        continue
                    lat = float(data.x[i, 0]) * 90.0
                    lon = float(data.x[i, 1]) * 180.0
                    if _haversine_km(lat, lon, cur["lat"], cur["lon"]) <= self.label_radius_km:
                        motion_target[i, 0] = (nxt["lat"] - cur["lat"]) / 10.0
                        motion_target[i, 1] = (nxt["lon"] - cur["lon"]) / 10.0
                        motion_mask[i] = True

        data.motion_target = motion_target
        data.motion_mask = motion_mask
        return data
