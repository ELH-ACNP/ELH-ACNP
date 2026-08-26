#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from algorithms.C_ALNS import execute_c_alns
from algorithms.ELH_ACNP import execute_pre_tuned_alns_cnp
from algorithms.GA_ELUMS import execute_ga_elums


ALGORITHMS = {
    "ELH-ACNP": lambda: execute_pre_tuned_alns_cnp(0.24),
    "C-ALNS": lambda: execute_c_alns(0.24),
    "GA-ELUMS": execute_ga_elums,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-start", type=int, default=100)
    parser.add_argument("--task-end", type=int, default=1500)
    parser.add_argument("--task-step", type=int, default=100)
    parser.add_argument("--satellites", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=tuple(ALGORITHMS),
        default=list(ALGORITHMS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "centralized_comparison.jsonl",
    )
    return parser.parse_args()


def read_completed(path: Path) -> set[tuple[int, int, str]]:
    completed: set[tuple[int, int, str]] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                completed.add(
                    (int(row["task_number"]), int(row["run"]), row["algorithm"])
                )
    return completed


def write_config(path: Path, config: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=4, sort_keys=True)


def append_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    args = parse_args()
    if args.task_start < 1 or args.task_end < args.task_start or args.task_step < 1:
        raise ValueError("Invalid task-scale range")
    if args.runs < 1:
        raise ValueError("runs must be positive")

    output_path = args.output.resolve()
    completed = read_completed(output_path)
    config_path = PROJECT_ROOT / "data" / "configure.json"
    with config_path.open("r", encoding="utf-8") as stream:
        original_config = json.load(stream)

    previous_task_seed = os.environ.get("SATELLITE_TASK_SEED")
    previous_algorithm_seed = os.environ.get("SATELLITE_ALGORITHM_SEED")
    try:
        for task_number in range(
            args.task_start, args.task_end + 1, args.task_step
        ):
            updated_config = dict(original_config)
            updated_config["Task_Number"] = task_number
            updated_config["Satellite_Number"] = args.satellites
            write_config(config_path, updated_config)

            for run_index in range(args.runs):
                task_seed = 10_000 + run_index
                algorithm_seed = 50_000 + run_index

                for algorithm_name in args.algorithms:
                    key = (task_number, run_index, algorithm_name)
                    if key in completed:
                        continue

                    os.environ["SATELLITE_TASK_SEED"] = str(task_seed)
                    os.environ["SATELLITE_ALGORITHM_SEED"] = str(algorithm_seed)
                    random.seed(algorithm_seed)
                    np.random.seed(algorithm_seed)

                    wall_start = time.perf_counter()
                    try:
                        reported_runtime, benefit, total_benefit = ALGORITHMS[
                            algorithm_name
                        ]()
                        record = {
                            "status": "ok",
                            "task_number": task_number,
                            "satellite_number": args.satellites,
                            "run": run_index,
                            "task_seed": task_seed,
                            "algorithm_seed": algorithm_seed,
                            "algorithm": algorithm_name,
                            "runtime_s": float(reported_runtime),
                            "wall_runtime_s": float(time.perf_counter() - wall_start),
                            "benefit": float(benefit),
                            "total_benefit": float(total_benefit),
                            "bcr": float(benefit / total_benefit),
                        }
                    except Exception as error:
                        record = {
                            "status": "error",
                            "task_number": task_number,
                            "satellite_number": args.satellites,
                            "run": run_index,
                            "task_seed": task_seed,
                            "algorithm_seed": algorithm_seed,
                            "algorithm": algorithm_name,
                            "wall_runtime_s": float(time.perf_counter() - wall_start),
                            "error": f"{type(error).__name__}: {error}",
                        }
                        append_record(output_path, record)
                        raise

                    append_record(output_path, record)
                    completed.add(key)
                    print(
                        f"tasks={task_number} run={run_index + 1}/{args.runs} "
                        f"algorithm={algorithm_name} bcr={record['bcr']:.6f} "
                        f"runtime={record['runtime_s']:.3f}s"
                    )
    finally:
        write_config(config_path, original_config)
        if previous_task_seed is None:
            os.environ.pop("SATELLITE_TASK_SEED", None)
        else:
            os.environ["SATELLITE_TASK_SEED"] = previous_task_seed
        if previous_algorithm_seed is None:
            os.environ.pop("SATELLITE_ALGORITHM_SEED", None)
        else:
            os.environ["SATELLITE_ALGORITHM_SEED"] = previous_algorithm_seed


if __name__ == "__main__":
    main()
