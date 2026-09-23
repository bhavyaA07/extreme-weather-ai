from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

from src.anomaly.detector import WeatherEvent


def _make_event(
    event_id: str,
    lead_time_idx: int,
    lat: float,
    lon: float,
    efi: float,
    variable: str = "precipitation",
) -> WeatherEvent:
    size = 1.0

    return WeatherEvent(
        event_id=event_id,
        lead_time_idx=lead_time_idx,
        mask=np.ones((3, 3), dtype=bool),
        centroid_latlon=(lat, lon),
        mean_efi=efi,
        max_efi=efi,
        area_km2=1000.0,
        variable=variable,
        bbox_latlon=(
            lat - size,
            lat + size,
            lon - size,
            lon + size,
        ),
    )


def load_training_runs(
    n_runs: int = 30,
    n_timesteps: int = 5,
    seed: int = 42,
) -> Tuple[List[List[List[WeatherEvent]]], List[List[dict]]]:
    """
    Generate deterministic synthetic forecast runs for testing the GNN pipeline.

    This is synthetic data only. It is intended to verify that the training
    pipeline works before replacing it with real ERA5/NWP + reference-track data.
    """

    random.seed(seed)

    forecast_runs = []
    reference_tracks = []

    for run_idx in range(n_runs):
        events_per_timestep = []
        track_points = []

        base_lat = 15.0 + random.uniform(-3.0, 3.0)
        base_lon = 70.0 + random.uniform(-5.0, 5.0)

        lat_step = random.uniform(0.3, 0.8)
        lon_step = random.uniform(0.4, 1.0)

        track_id = f"storm_{run_idx}"

        for t in range(n_timesteps):
            storm_lat = base_lat + t * lat_step
            storm_lon = base_lon + t * lon_step

            timestep_events = []

            main_event = _make_event(
                event_id=f"run{run_idx}_storm_t{t}",
                lead_time_idx=t,
                lat=storm_lat + random.uniform(-0.15, 0.15),
                lon=storm_lon + random.uniform(-0.15, 0.15),
                efi=2.5 + random.uniform(0.0, 1.0),
            )
            timestep_events.append(main_event)

            noise_event = _make_event(
                event_id=f"run{run_idx}_noise_t{t}",
                lead_time_idx=t,
                lat=storm_lat + random.uniform(4.0, 8.0),
                lon=storm_lon + random.uniform(4.0, 8.0),
                efi=random.uniform(2.0, 3.0),
            )
            timestep_events.append(noise_event)

            events_per_timestep.append(timestep_events)

            track_points.append(
                {
                    "lead_time_idx": t,
                    "lat": storm_lat,
                    "lon": storm_lon,
                    "track_id": track_id,
                }
            )

        forecast_runs.append(events_per_timestep)
        reference_tracks.append(track_points)

    return forecast_runs, reference_tracks
