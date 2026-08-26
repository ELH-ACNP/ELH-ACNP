#!/usr/bin/env python3
"""Synthetic task locations and GEO DRS visibility calculations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


EARTH_RADIUS_KM = 6378.137
GEO_ORBIT_RADIUS_KM = 42164.0


def load_visibility_settings(config_path: str | Path) -> dict[str, object]:
    """Read and validate the reproducible visibility settings."""
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    satellite_number = int(config["Satellite_Number"])
    longitudes = np.asarray(config["Satellite_Longitudes_Deg"], dtype=float)
    if satellite_number < 1 or satellite_number > longitudes.size:
        raise ValueError(
            "Satellite_Number must be between 1 and the number of configured "
            "satellite longitudes"
        )

    location = config["Task_Location_Distribute"]
    settings = {
        "satellite_longitudes_deg": longitudes[:satellite_number],
        "minimum_elevation_deg": float(config["Minimum_Elevation_Deg"]),
        "longitude_mean_deg": float(location["Longitude_Mean_Deg"]),
        "longitude_std_deg": float(location["Longitude_SD_Deg"]),
        "longitude_min_deg": float(location["Longitude_Min_Deg"]),
        "longitude_max_deg": float(location["Longitude_Max_Deg"]),
        "latitude_mean_deg": float(location["Latitude_Mean_Deg"]),
        "latitude_std_deg": float(location["Latitude_SD_Deg"]),
        "latitude_min_deg": float(location["Latitude_Min_Deg"]),
        "latitude_max_deg": float(location["Latitude_Max_Deg"]),
    }
    return settings


def _truncated_normal(
    rng: np.random.RandomState,
    mean: float,
    standard_deviation: float,
    lower_bound: float,
    upper_bound: float,
    size: int,
) -> np.ndarray:
    if standard_deviation <= 0 or lower_bound >= upper_bound:
        raise ValueError("Invalid truncated-normal parameters")

    values = rng.normal(mean, standard_deviation, size)
    invalid = (values < lower_bound) | (values > upper_bound)
    while np.any(invalid):
        values[invalid] = rng.normal(mean, standard_deviation, int(np.sum(invalid)))
        invalid = (values < lower_bound) | (values > upper_bound)
    return values


def generate_task_locations(
    task_number: int,
    task_seed: int,
    settings: dict[str, object],
) -> np.ndarray:
    """Generate reproducible synthetic longitude-latitude task locations."""
    # A separate stream prevents location generation from changing the task
    # windows, durations, bandwidths, and benefits produced by the main stream.
    rng = np.random.RandomState(task_seed + 1_000_003)
    longitude = _truncated_normal(
        rng,
        float(settings["longitude_mean_deg"]),
        float(settings["longitude_std_deg"]),
        float(settings["longitude_min_deg"]),
        float(settings["longitude_max_deg"]),
        task_number,
    )
    latitude = _truncated_normal(
        rng,
        float(settings["latitude_mean_deg"]),
        float(settings["latitude_std_deg"]),
        float(settings["latitude_min_deg"]),
        float(settings["latitude_max_deg"]),
        task_number,
    )
    return np.column_stack((longitude, latitude))


def calculate_elevation_angles(
    task_locations: np.ndarray,
    satellite_longitudes_deg: np.ndarray,
) -> np.ndarray:
    """Calculate ground-to-GEO elevation angles in degrees."""
    longitude = np.deg2rad(task_locations[:, 0])[:, None]
    latitude = np.deg2rad(task_locations[:, 1])[:, None]
    satellite_longitude = np.deg2rad(satellite_longitudes_deg)[None, :]

    central_cosine = np.cos(latitude) * np.cos(longitude - satellite_longitude)
    central_cosine = np.clip(central_cosine, -1.0, 1.0)
    numerator = central_cosine - EARTH_RADIUS_KM / GEO_ORBIT_RADIUS_KM
    denominator = np.sqrt(np.maximum(0.0, 1.0 - central_cosine**2))
    return np.rad2deg(np.arctan2(numerator, denominator))


def generate_visibility_windows(
    task_locations: np.ndarray,
    satellite_longitudes_deg: np.ndarray,
    minimum_elevation_deg: float,
    scheduling_start: float,
    scheduling_end: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return elevation angles, eligibility, and per-DRS visibility windows.

    GEO visibility is stationary over the one-day simulation. A visible pair
    therefore receives the full scheduling horizon as its visibility window;
    an invisible pair receives NaN bounds and cannot submit a bid.
    """
    elevation_angles = calculate_elevation_angles(
        task_locations,
        satellite_longitudes_deg,
    )
    visibility = elevation_angles >= minimum_elevation_deg
    if np.any(np.sum(visibility, axis=1) == 0):
        raise ValueError("At least one generated task is invisible to every DRS")

    windows = np.full((*visibility.shape, 2), np.nan, dtype=float)
    windows[visibility, 0] = scheduling_start
    windows[visibility, 1] = scheduling_end
    return elevation_angles, visibility, windows


def build_parameter_packages(
    task_index: int,
    tasklist: np.ndarray,
    visibility_windows: np.ndarray,
    satellites: list,
    potential_conflicts: set,
    time_list: list,
    bandwidth: float,
) -> list[list[object]]:
    """Build bid packets only for DRSs with a valid visibility window."""
    task_start, task_end, task_width, task_duration = tasklist[task_index, :4]
    packages: list[list[object]] = []

    for satellite in satellites:
        satellite_id = satellite.satellite_id
        visibility_start, visibility_end = visibility_windows[task_index, satellite_id]
        if not np.isfinite(visibility_start) or not np.isfinite(visibility_end):
            continue

        request_start = max(float(task_start), float(visibility_start))
        request_end = min(float(task_end), float(visibility_end))
        if request_end - request_start < float(task_duration):
            continue

        conflict_set = potential_conflicts & satellite.execution_list
        occupied_windows = [
            [
                time_list[satellite_id][conflict_task],
                tasklist[conflict_task, 3],
                tasklist[conflict_task, 2],
            ]
            for conflict_task in conflict_set
        ]
        packages.append([
            bandwidth,
            [request_start, request_end],
            task_duration,
            task_width,
            occupied_windows,
            satellite_id,
        ])

    return packages
