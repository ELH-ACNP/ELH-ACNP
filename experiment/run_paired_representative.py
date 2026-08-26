#!/usr/bin/env python3
"""Run paired representative-scenario experiments with resumable checkpoints."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(Path(__file__).resolve().parent)

from algorithms.ELH_ACNP import execute_pre_tuned_alns_cnp
from algorithms.LNS_CNP import execute_lns_cnp
from algorithms.VNS_CNP import execute_vns_cnp
from algorithms.VND_CNP import execute_vnd_cnp
from algorithms.HCNP import execute_ns_cnp
from algorithms.HPFS_CNP import execute_hpfs_cnp
from algorithms.FCFS_CNP import execute_fcfs_cnp
from algorithms.LLF_CNP import execute_llf_cnp


SCENARIOS = {
    "section_5_3": {
        "task_number": 1500,
        "satellite_number": 5,
        "description": "large-scale task scenario",
    },
    "section_5_4": {
        "task_number": 1500,
        "satellite_number": 3,
        "description": "resource-constrained multi-DRS scenario",
    },
}

ALGORITHMS = {
    "ELH-ACNP": lambda: execute_pre_tuned_alns_cnp(0.24),
    "LNS-CNP": lambda: execute_lns_cnp(0.24),
    "VNS-CNP": lambda: execute_vns_cnp(0.24),
    "VND-CNP": lambda: execute_vnd_cnp(0.24),
    "H-CNP": lambda: execute_ns_cnp(0.24),
    "HPFS-CNP": execute_hpfs_cnp,
    "FCFS-CNP": execute_fcfs_cnp,
    "LLF-CNP": execute_llf_cnp,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_completed(path: Path) -> set[tuple[str, int, str]]:
    completed: set[tuple[str, int, str]] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                completed.add((row["scenario"], int(row["run"]), row["algorithm"]))
    return completed


def update_config(config_path: Path, scenario: dict[str, object]) -> dict[str, object]:
    with config_path.open("r", encoding="utf-8") as stream:
        original = json.load(stream)
    updated = dict(original)
    updated["Task_Number"] = int(scenario["task_number"])
    updated["Satellite_Number"] = int(scenario["satellite_number"])
    with config_path.open("w", encoding="utf-8") as stream:
        json.dump(updated, stream, indent=4, sort_keys=True)
    return original


def write_config(config_path: Path, config: dict[str, object]) -> None:
    with config_path.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=4, sort_keys=True)


def append_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    args = parse_args()
    scenario = SCENARIOS[args.scenario]
    output_path = args.output.resolve()
    completed = read_completed(output_path)

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "data" / "configure.json"
    original_config = update_config(config_path, scenario)

    try:
        for run_index in range(args.runs):
            task_seed = 10_000 + run_index
            algorithm_seed = 50_000 + run_index
            algorithm_order = list(ALGORITHMS)
            random.Random(90_000 + run_index).shuffle(algorithm_order)

            for algorithm_name in algorithm_order:
                key = (args.scenario, run_index, algorithm_name)
                if key in completed:
                    continue

                os.environ["SATELLITE_TASK_SEED"] = str(task_seed)
                os.environ["SATELLITE_ALGORITHM_SEED"] = str(algorithm_seed)
                random.seed(algorithm_seed)
                np.random.seed(algorithm_seed)

                captured_stdout = io.StringIO()
                captured_stderr = io.StringIO()
                wall_start = time.perf_counter()
                try:
                    with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
                        reported_runtime, benefit, total_benefit = ALGORITHMS[algorithm_name]()
                    wall_runtime = time.perf_counter() - wall_start
                    record = {
                        "status": "ok",
                        "scenario": args.scenario,
                        "scenario_description": scenario["description"],
                        "task_number": int(scenario["task_number"]),
                        "satellite_number": int(scenario["satellite_number"]),
                        "run": run_index,
                        "task_seed": task_seed,
                        "algorithm_seed": algorithm_seed,
                        "algorithm": algorithm_name,
                        "benefit": float(benefit),
                        "total_benefit": float(total_benefit),
                        "bcr": float(benefit / total_benefit),
                        "runtime_reported_s": float(reported_runtime),
                        "runtime_wall_s": float(wall_runtime),
                    }
                except Exception as exc:
                    wall_runtime = time.perf_counter() - wall_start
                    record = {
                        "status": "error",
                        "scenario": args.scenario,
                        "run": run_index,
                        "task_seed": task_seed,
                        "algorithm_seed": algorithm_seed,
                        "algorithm": algorithm_name,
                        "runtime_wall_s": float(wall_runtime),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "captured_stdout": captured_stdout.getvalue()[-4000:],
                        "captured_stderr": captured_stderr.getvalue()[-4000:],
                    }
                    append_record(output_path, record)
                    raise

                append_record(output_path, record)
                completed.add(key)
                print(
                    f"{args.scenario} run={run_index + 1:02d}/{args.runs} "
                    f"algorithm={algorithm_name:9s} BCR={record['bcr']:.6f} "
                    f"runtime={record['runtime_reported_s']:.3f}s wall={record['runtime_wall_s']:.3f}s",
                    flush=True,
                )
    finally:
        write_config(config_path, original_config)


if __name__ == "__main__":
    main()
