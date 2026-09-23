"""
Builds a spatio-temporal graph out of WeatherEvent detections so a GNN can
learn which event at time t is "the same storm" as which event at time t+1
(association / tracking), and can propagate a predicted future track.

Node  = one detected event at one lead time (features: centroid, EFI, area, variable)
Edge  = candidate link between events at consecutive lead times that are close
        enough in space to plausibly be the same evolving system.

The GNN (see model.py) scores each candidate edge; edges above a threshold
are kept as the storm's track.
"""

from __future__ import annotations
from typing import List
import numpy as np
import torch
from torch_geometric.data import Data

from src.anomaly.detector import WeatherEvent


NODE_FEATURES = ["lat", "lon", "mean_efi", "max_efi", "area_km2_norm", "lead_time_idx"]


def _event_to_features(ev: WeatherEvent, max_area_km2: float = 500_000.0) -> np.ndarray:
    lat, lon = ev.centroid_latlon
    return np.array(
        [
            lat / 90.0,
            lon / 180.0,
            ev.mean_efi,
            ev.max_efi,
            min(ev.area_km2 / max_area_km2, 1.0),
            ev.lead_time_idx / 10.0,  # normalise assuming ~10 lead times
        ],
        dtype=np.float32,
    )


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_track_graph(
    events_per_timestep: List[List[WeatherEvent]],
    max_link_distance_km: float = 300.0,
) -> Data:
    """
    events_per_timestep[t] = list of WeatherEvent detected at lead time t.

    Returns a torch_geometric Data object with:
      x           : (N, F) node features
      edge_index  : (2, E) candidate temporal edges (t -> t+1, within max_link_distance_km)
      edge_attr   : (E, 2) [distance_km_norm, efi_similarity]
      node_meta   : list of (event_id, lead_time_idx) for mapping predictions back to events
    """
    flat_events: List[WeatherEvent] = []
    node_id_by_index: dict = {}
    for t, events in enumerate(events_per_timestep):
        for ev in events:
            node_id_by_index[len(flat_events)] = ev
            flat_events.append(ev)

    if not flat_events:
        return Data(x=torch.zeros((0, len(NODE_FEATURES))), edge_index=torch.zeros((2, 0), dtype=torch.long))

    x = torch.tensor(np.stack([_event_to_features(ev) for ev in flat_events]), dtype=torch.float)

    src, dst, edge_feats = [], [], []
    offset_by_t = {}
    idx = 0
    for t, events in enumerate(events_per_timestep):
        offset_by_t[t] = idx
        idx += len(events)

    for t in range(len(events_per_timestep) - 1):
        cur_events = events_per_timestep[t]
        next_events = events_per_timestep[t + 1]
        for i, ev_a in enumerate(cur_events):
            for j, ev_b in enumerate(next_events):
                dist = _haversine_km(*ev_a.centroid_latlon, *ev_b.centroid_latlon)
                if dist <= max_link_distance_km:
                    a_idx = offset_by_t[t] + i
                    b_idx = offset_by_t[t + 1] + j
                    src.append(a_idx)
                    dst.append(b_idx)
                    efi_sim = 1.0 - min(abs(ev_a.mean_efi - ev_b.mean_efi), 1.0)
                    edge_feats.append([dist / max_link_distance_km, efi_sim])

    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(edge_feats, dtype=torch.float) if edge_feats else torch.zeros((0, 2))

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_meta = [(ev.event_id, ev.lead_time_idx) for ev in flat_events]
    return data
