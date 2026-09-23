"""
Converts a WeatherEvent (coarse-grid footprint) into a set of fine-grained
(~5 km) alert zones once the downscaling model has produced a high-res field
inside the event's bounding box.

Two-stage zoning:
  1. Coarse bbox from the detector (12 km grid) — used to crop the region
     that gets sent to the downscaling model.
  2. Fine 5 km zone polygons — computed from the downscaled field, so the
     alert footprint reflects true sub-grid intensity, not just the coarse box.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class AlertZone:
    zone_id: str
    lat: float
    lon: float
    severity: str          # "low" | "moderate" | "severe"
    variable: str
    value: float            # downscaled physical value (mm, km/h, etc.)
    lead_time_idx: int


def pad_bbox(bbox_latlon: tuple, pad_deg: float = 0.15) -> tuple:
    """Expand a coarse bbox slightly so the downscaling model has context at the edges."""
    lat_min, lat_max, lon_min, lon_max = bbox_latlon
    return (lat_min - pad_deg, lat_max + pad_deg, lon_min - pad_deg, lon_max + pad_deg)


def severity_from_value(value: float, thresholds: dict) -> str:
    """
    thresholds e.g. {"moderate": 50, "severe": 100}  (mm of rain per 24h, or km/h wind, etc.)
    """
    if value >= thresholds["severe"]:
        return "severe"
    if value >= thresholds["moderate"]:
        return "moderate"
    return "low"


def build_alert_zones(
    downscaled_field: np.ndarray,
    lat_grid_fine: np.ndarray,
    lon_grid_fine: np.ndarray,
    variable: str,
    lead_time_idx: int,
    severity_thresholds: dict,
    fine_cell_km: float = 5.0,
    min_value: float = 0.0,
) -> List[AlertZone]:
    """
    Turns a downscaled (fine-grid) field for one event's bounding box into a
    flat list of ~5 km alert zones, each tagged with severity.
    """
    zones: List[AlertZone] = []
    H, W = downscaled_field.shape
    for i in range(H):
        for j in range(W):
            value = float(downscaled_field[i, j])
            if value <= min_value:
                continue
            zones.append(
                AlertZone(
                    zone_id=f"{variable}_t{lead_time_idx}_{i}_{j}",
                    lat=float(lat_grid_fine[i, j]),
                    lon=float(lon_grid_fine[i, j]),
                    severity=severity_from_value(value, severity_thresholds),
                    variable=variable,
                    value=value,
                    lead_time_idx=lead_time_idx,
                )
            )
    return zones
