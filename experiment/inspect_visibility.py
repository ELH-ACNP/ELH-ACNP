#!/usr/bin/env python3
"""Inspect synthetic task locations and GEO DRS visibility statistics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from task_tool.HelpLibrary import Helper
from task_tool.Task_generator import Task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = PROJECT_ROOT / "data" / "configure.json"
    task_info, scene_info, _ = Helper(str(config_path)).parse_world_info()
    os.environ["SATELLITE_TASK_SEED"] = str(args.seed)

    task = Task(
        args.tasks,
        scene_info[0],
        scene_info[1],
        *task_info[1:],
        config_dir=str(config_path),
    )
    visible_per_task = np.sum(task.visibility_matrix, axis=1)
    summary = {
        "task_seed": args.seed,
        "task_number": args.tasks,
        "satellite_longitudes_deg": task.satellite_longitudes.tolist(),
        "overall_visible_pair_rate": float(np.mean(task.visibility_matrix)),
        "tasks_visible_to_all_drs_rate": float(
            np.mean(visible_per_task == task.visibility_matrix.shape[1])
        ),
        "minimum_visible_drs_per_task": int(np.min(visible_per_task)),
        "minimum_elevation_deg": float(np.min(task.elevation_angles)),
        "maximum_elevation_deg": float(np.max(task.elevation_angles)),
        "generated_longitude_range_deg": [
            float(np.min(task.task_locations[:, 0])),
            float(np.max(task.task_locations[:, 0])),
        ],
        "generated_latitude_range_deg": [
            float(np.min(task.task_locations[:, 1])),
            float(np.max(task.task_locations[:, 1])),
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
