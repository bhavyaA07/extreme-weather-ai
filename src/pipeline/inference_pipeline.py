"""
End-to-end pipeline: coarse forecast -> EFI -> events -> GNN tracks ->
downscaled 5km fields -> alert zones with severity.

This is the file your backend (backend/main.py) calls per new forecast run.
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch

from src.anomaly.efi import compute_efi_timeseries
from src.anomaly.detector import AnomalyDetector, WeatherEvent
from src.anomaly.bounding_box import pad_bbox, build_alert_zones, AlertZone
from src.gnn.tracker import StormTracker
from src.downscaling.inference import Downscaler

SEVERITY_THRESHOLDS = {
    "precipitation": {"moderate": 50, "severe": 100},   # mm / 24h
    "wind": {"moderate": 60, "severe": 100},             # km/h
    "temperature": {"moderate": 44, "severe": 47},       # deg C
}


class ExtremeWeatherPipeline:
    def __init__(
        self,
        gnn_model_path: str,
        diffusion_model_path: str,
        efi_threshold: float = 0.5,
        edge_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.detector = AnomalyDetector(efi_threshold=efi_threshold)
        self.tracker = StormTracker(gnn_model_path, device=device)
        self.downscaler = Downscaler(diffusion_model_path, device=device)
        self.edge_threshold = edge_threshold

    def run(
        self,
        forecast_sequence: np.ndarray,       # (T, H, W) coarse forecast, one variable
        climatology_samples: np.ndarray,     # (N, H, W) climatology for that variable/season
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        topography_fine: Dict[str, np.ndarray],  # {"topo": ..., "lsm": ...} at 5km resolution, full domain
        variable: str = "precipitation",
    ) -> Dict:
        # 1. EFI
        efi_seq = compute_efi_timeseries(forecast_sequence, climatology_samples)

        # 2. Detection per lead time
        events_per_timestep: List[List[WeatherEvent]] = self.detector.detect_sequence(
            efi_seq, lat_grid, lon_grid, variable=variable
        )

        # 3. Tracking across lead times
        tracks = self.tracker.track(events_per_timestep, edge_threshold=self.edge_threshold)

        # 4. Downscale + alert-zone each event, keyed by event_id
        events_by_id = {ev.event_id: ev for evs in events_per_timestep for ev in evs}
        all_zones: List[AlertZone] = []
        for track in tracks:
            for event_id in track:
                ev = events_by_id.get(event_id)
                if ev is None:
                    continue
                fine_field = self._downscale_event(ev, forecast_sequence, topography_fine, lat_grid, lon_grid)
                if fine_field is None:
                    continue
                fine_lat, fine_lon = fine_field["lat_grid"], fine_field["lon_grid"]
                zones = build_alert_zones(
                    downscaled_field=fine_field["value"],
                    lat_grid_fine=fine_lat,
                    lon_grid_fine=fine_lon,
                    variable=variable,
                    lead_time_idx=ev.lead_time_idx,
                    severity_thresholds=SEVERITY_THRESHOLDS.get(variable, {"moderate": 50, "severe": 100}),
                )
                all_zones.extend(zones)

        return {
            "tracks": tracks,
            "events": {eid: ev for eid, ev in events_by_id.items()},
            "alert_zones": all_zones,
        }

    def _downscale_event(self, ev: WeatherEvent, forecast_sequence, topography_fine, lat_grid, lon_grid):
        """
        Crops the coarse forecast + fine topography to the (padded) event bbox
        and runs the diffusion downscaler. Placeholder cropping logic — wire up
        to your actual grid indexing utilities in src/utils/geo.py.
        """
        bbox = pad_bbox(ev.bbox_latlon)
        # TODO: replace with real lat/lon -> index cropping via src/utils/geo.py
        # coarse_patch, topo_patch, lsm_patch, fine_lat, fine_lon = crop_to_bbox(...)
        # fine_field = self.downscaler.downscale_patch(coarse_patch, topo_patch, lsm_patch, target_hw)
        # return {"value": fine_field.cpu().numpy(), "lat_grid": fine_lat, "lon_grid": fine_lon}
        return None
