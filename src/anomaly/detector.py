"""
Turns a per-cell EFI field into discrete "extreme weather event" objects
(connected regions of anomalous cells), one set per lead time.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np
from scipy import ndimage


@dataclass
class WeatherEvent:
    event_id: str
    lead_time_idx: int
    mask: np.ndarray            # boolean (H, W) footprint on the model grid
    centroid_latlon: tuple       # (lat, lon)
    mean_efi: float
    max_efi: float
    area_km2: float
    variable: str
    bbox_latlon: tuple = field(default=None)  # (lat_min, lat_max, lon_min, lon_max)


class AnomalyDetector:
    """
    Thresholds EFI, groups connected anomalous cells into events, and
    computes summary stats per event. Designed to run per variable
    (rain, wind, temperature) and be merged downstream.
    """

    def __init__(
        self,
        efi_threshold: float = 0.5,
        min_cells: int = 3,
        cell_size_km: float = 12.0,
        connectivity: int = 2,  # 1 = 4-connectivity, 2 = 8-connectivity
    ):
        self.efi_threshold = efi_threshold
        self.min_cells = min_cells
        self.cell_size_km = cell_size_km
        self.structure = ndimage.generate_binary_structure(2, connectivity)

    def detect(
        self,
        efi_field: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        lead_time_idx: int,
        variable: str = "precipitation",
    ) -> List[WeatherEvent]:
        """
        efi_field : (H, W) EFI values for this lead time
        lat_grid, lon_grid : (H, W) coordinate grids matching efi_field
        """
        anomaly_mask = np.abs(efi_field) >= self.efi_threshold
        labeled, n_labels = ndimage.label(anomaly_mask, structure=self.structure)

        events: List[WeatherEvent] = []
        for label_id in range(1, n_labels + 1):
            region_mask = labeled == label_id
            n_cells = int(region_mask.sum())
            if n_cells < self.min_cells:
                continue

            region_efi = efi_field[region_mask]
            lats = lat_grid[region_mask]
            lons = lon_grid[region_mask]

            events.append(
                WeatherEvent(
                    event_id=f"{variable}_t{lead_time_idx}_{label_id}",
                    lead_time_idx=lead_time_idx,
                    mask=region_mask,
                    centroid_latlon=(float(lats.mean()), float(lons.mean())),
                    mean_efi=float(region_efi.mean()),
                    max_efi=float(region_efi[np.argmax(np.abs(region_efi))]),
                    area_km2=n_cells * (self.cell_size_km ** 2),
                    variable=variable,
                    bbox_latlon=(
                        float(lats.min()), float(lats.max()),
                        float(lons.min()), float(lons.max()),
                    ),
                )
            )
        return events

    def detect_sequence(
        self,
        efi_sequence: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        variable: str = "precipitation",
    ) -> List[List[WeatherEvent]]:
        """Run detect() across all lead times. Returns events_per_timestep[T]."""
        return [
            self.detect(efi_sequence[t], lat_grid, lon_grid, t, variable)
            for t in range(efi_sequence.shape[0])
        ]
