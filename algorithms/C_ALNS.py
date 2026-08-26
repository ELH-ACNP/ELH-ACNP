#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import time

import numpy as np

from task_tool.HelpLibrary import Helper
from task_tool.Task_generator import Task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "data" / "configure.json"


@dataclass
class CentralizedSchedule:
    """Decoded centralized schedule."""

    benefit: float
    task_to_satellite: np.ndarray
    start_times: np.ndarray
    frequency_starts: np.ndarray


class CentralizedDecoder:
    """Decode a task permutation into one feasible multi-DRS schedule."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG,
        task_number_override: int | None = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        task_info, scene_info, resource_info = Helper(
            str(self.config_path)
        ).parse_world_info()

        task_number = (
            int(task_number_override)
            if task_number_override is not None
            else int(task_info[0])
        )
        task = Task(
            task_number,
            scene_info[0],
            scene_info[1],
            *task_info[1:],
            config_dir=str(self.config_path),
        )

        self.tasklist = np.asarray(task.tasklist, dtype=float)
        self.conflict_sets = task.conflict_set
        self.visibility_windows = np.asarray(task.visibility_windows, dtype=float)
        self.satellite_longitudes = np.asarray(task.satellite_longitudes, dtype=float)
        self.task_number = task_number
        self.satellite_number = int(resource_info[0])
        self.bandwidth = float(resource_info[-1])
        self.total_benefit = float(np.sum(self.tasklist[:, 4]))

    @staticmethod
    def _find_time_slice(
        bandwidth: float,
        request_window: tuple[float, float],
        duration: float,
        required_bandwidth: float,
        occupied_windows: list[tuple[float, float, float]],
    ) -> tuple[float, float] | None:
        """Return the earliest feasible start and minimum free bandwidth."""
        request_start, request_end = request_window
        if request_end - request_start + 1e-12 < duration:
            return None
        if not occupied_windows:
            return request_start, bandwidth

        events: dict[float, float] = {request_start: 0.0, request_end: 0.0}
        for occupied_start, occupied_duration, occupied_width in occupied_windows:
            occupied_end = occupied_start + occupied_duration
            overlap_start = max(request_start, occupied_start)
            overlap_end = min(request_end, occupied_end)
            if overlap_start < overlap_end:
                events[overlap_start] = events.get(overlap_start, 0.0) + occupied_width
                events[overlap_end] = events.get(overlap_end, 0.0) - occupied_width

        used_bandwidth = 0.0
        feasible_start: float | None = None
        minimum_free = bandwidth
        event_times = sorted(events)
        for index, event_time in enumerate(event_times[:-1]):
            used_bandwidth += events[event_time]
            next_time = event_times[index + 1]
            free_bandwidth = bandwidth - used_bandwidth

            if free_bandwidth + 1e-12 >= required_bandwidth:
                if feasible_start is None:
                    feasible_start = event_time
                    minimum_free = free_bandwidth
                else:
                    minimum_free = min(minimum_free, free_bandwidth)
                if next_time - feasible_start + 1e-12 >= duration:
                    return feasible_start, minimum_free
            else:
                feasible_start = None
                minimum_free = bandwidth
        return None

    def decode(self, task_sequence: np.ndarray | list[int]) -> CentralizedSchedule:
        """Schedule tasks in the supplied centralized priority order."""
        sequence = np.asarray(task_sequence, dtype=int)
        if sequence.shape != (self.task_number,):
            raise ValueError("A chromosome must contain every task exactly once")
        if np.unique(sequence).size != self.task_number:
            raise ValueError("A chromosome contains duplicate or missing tasks")
        if np.min(sequence) != 0 or np.max(sequence) != self.task_number - 1:
            raise ValueError("Task indices are outside the valid range")

        assigned_tasks = [set() for _ in range(self.satellite_number)]
        task_to_satellite = np.full(self.task_number, -1, dtype=int)
        start_times = np.full(self.task_number, np.nan, dtype=float)
        frequency_starts = np.full(self.task_number, np.nan, dtype=float)
        benefit = 0.0

        for task_index in sequence:
            task_index = int(task_index)
            task_start, task_end, task_width, task_duration = self.tasklist[
                task_index, :4
            ]
            best_candidate: tuple[float, int, float] | None = None

            for satellite_id in range(self.satellite_number):
                visibility_start, visibility_end = self.visibility_windows[
                    task_index, satellite_id
                ]
                if not np.isfinite(visibility_start) or not np.isfinite(visibility_end):
                    continue

                request_start = max(float(task_start), float(visibility_start))
                request_end = min(float(task_end), float(visibility_end))
                conflicts = self.conflict_sets[task_index] & assigned_tasks[satellite_id]
                occupied_windows = [
                    (
                        float(start_times[conflict]),
                        float(self.tasklist[conflict, 3]),
                        float(self.tasklist[conflict, 2]),
                    )
                    for conflict in conflicts
                ]
                feasible = self._find_time_slice(
                    self.bandwidth,
                    (request_start, request_end),
                    float(task_duration),
                    float(task_width),
                    occupied_windows,
                )
                if feasible is None:
                    continue

                feasible_start, minimum_free = feasible
                candidate = (feasible_start, satellite_id, minimum_free)
                if best_candidate is None or candidate[:2] < best_candidate[:2]:
                    best_candidate = candidate

            if best_candidate is None:
                continue

            selected_start, selected_satellite, minimum_free = best_candidate
            assigned_tasks[selected_satellite].add(task_index)
            task_to_satellite[task_index] = selected_satellite
            start_times[task_index] = selected_start
            frequency_starts[task_index] = self.bandwidth - minimum_free
            benefit += float(self.tasklist[task_index, 4])

        return CentralizedSchedule(
            benefit=benefit,
            task_to_satellite=task_to_satellite,
            start_times=start_times,
            frequency_starts=frequency_starts,
        )

    def validate(self, schedule: CentralizedSchedule) -> None:
        """Raise an error if a decoded schedule violates model constraints."""
        selected = np.flatnonzero(schedule.task_to_satellite >= 0)
        calculated_benefit = float(np.sum(self.tasklist[selected, 4]))
        if not np.isclose(calculated_benefit, schedule.benefit):
            raise ValueError("Schedule benefit is inconsistent with selected tasks")

        for task_index in selected:
            satellite_id = int(schedule.task_to_satellite[task_index])
            start = float(schedule.start_times[task_index])
            duration = float(self.tasklist[task_index, 3])
            service_start, service_end = self.tasklist[task_index, :2]
            visibility_start, visibility_end = self.visibility_windows[
                task_index, satellite_id
            ]
            if start + 1e-12 < max(service_start, visibility_start):
                raise ValueError("A task starts before its feasible window")
            if start + duration > min(service_end, visibility_end) + 1e-12:
                raise ValueError("A task ends after its feasible window")

        for satellite_id in range(self.satellite_number):
            events: list[tuple[float, float]] = []
            satellite_tasks = selected[
                schedule.task_to_satellite[selected] == satellite_id
            ]
            for task_index in satellite_tasks:
                start = float(schedule.start_times[task_index])
                duration = float(self.tasklist[task_index, 3])
                width = float(self.tasklist[task_index, 2])
                events.append((start, width))
                events.append((start + duration, -width))

            occupied = 0.0
            for _, change in sorted(events, key=lambda event: (event[0], event[1])):
                occupied += change
                if occupied > self.bandwidth + 1e-9:
                    raise ValueError("A DRS exceeds its available bandwidth")
                if occupied < -1e-9:
                    raise ValueError("Invalid bandwidth event sequence")


class CentralizedALNS:
    """Adaptive large-neighborhood search over centralized task sequences."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG,
        max_iterations: int = 100,
        max_non_improving: int = 100,
        disposal_rate: float = 0.24,
        initial_temperature: float = 100.0,
        minimum_temperature: float = 0.01,
        cooling_rate: float = 0.95,
        update_interval: int = 10,
        reaction_factor: float = 0.2,
        task_number_override: int | None = None,
    ) -> None:
        if max_iterations < 1 or max_non_improving < 1:
            raise ValueError("Iteration limits must be positive")
        if not 0 < disposal_rate <= 1:
            raise ValueError("disposal_rate must be in (0, 1]")

        self.decoder = CentralizedDecoder(config_path, task_number_override)
        self.max_iterations = int(max_iterations)
        self.max_non_improving = int(max_non_improving)
        self.disposal_rate = float(disposal_rate)
        self.initial_temperature = float(initial_temperature)
        self.minimum_temperature = float(minimum_temperature)
        self.cooling_rate = float(cooling_rate)
        self.update_interval = int(update_interval)
        self.reaction_factor = float(reaction_factor)
        self.rng = np.random.default_rng(
            int(os.environ.get("SATELLITE_ALGORITHM_SEED", "3845"))
        )

        self.operator_weights = np.full((4, 4), 1 / 16, dtype=float)
        self.operator_scores = np.zeros((4, 4), dtype=float)
        self.operator_calls = np.zeros((4, 4), dtype=int)
        self.best_schedule: CentralizedSchedule | None = None
        self.best_sequence: np.ndarray | None = None
        self.benefit_curve: list[float] = []
        self.time = 0.0

        tasklist = self.decoder.tasklist
        self._destroy_metrics = (
            None,
            tasklist[:, 4],
            np.asarray([len(item) for item in self.decoder.conflict_sets], dtype=float),
            tasklist[:, 5],
        )
        self._repair_orders = (
            None,
            np.argsort(-tasklist[:, 4]),
            np.argsort(np.asarray([len(item) for item in self.decoder.conflict_sets])),
            np.argsort(-tasklist[:, 5]),
        )
        self._repair_ranks = [None]
        for order in self._repair_orders[1:]:
            ranks = np.empty(self.decoder.task_number, dtype=int)
            ranks[order] = np.arange(self.decoder.task_number)
            self._repair_ranks.append(ranks)

    @staticmethod
    def _normalise(values: np.ndarray) -> np.ndarray:
        span = float(np.max(values) - np.min(values))
        if span <= 0:
            return np.zeros_like(values, dtype=float)
        return (values - np.min(values)) / span

    def _construct_initial_sequence(self) -> np.ndarray:
        """Stage one: construct a centralized priority sequence."""
        tasklist = self.decoder.tasklist
        benefit = self._normalise(tasklist[:, 4])
        efficiency = self._normalise(tasklist[:, 5])
        conflict = self._normalise(
            np.asarray([len(item) for item in self.decoder.conflict_sets], dtype=float)
        )
        slack = (tasklist[:, 1] - tasklist[:, 0]) - tasklist[:, 3]
        urgency = 1.0 - self._normalise(slack)
        score = 0.4 * benefit + 0.3 * efficiency + 0.2 * urgency + 0.1 * (1 - conflict)
        return np.lexsort((np.arange(self.decoder.task_number), -score)).astype(int)

    def _select_operator_pair(self) -> tuple[int, int]:
        probabilities = self.operator_weights.ravel()
        probabilities = probabilities / probabilities.sum()
        pair_index = int(self.rng.choice(probabilities.size, p=probabilities))
        return tuple(int(item) for item in np.unravel_index(pair_index, (4, 4)))

    def _destroy(
        self,
        sequence: np.ndarray,
        operator_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        remove_count = min(
            sequence.size - 1,
            max(1, int(math.ceil(self.disposal_rate * sequence.size))),
        )
        if operator_index == 0:
            removed = self.rng.choice(sequence, remove_count, replace=False)
        else:
            metric = self._destroy_metrics[operator_index]
            if operator_index in (1, 3):
                order = sequence[np.argsort(metric[sequence])]
            else:
                order = sequence[np.argsort(-metric[sequence])]
            pool_size = min(sequence.size, max(remove_count, 4 * remove_count))
            removed = self.rng.choice(order[:pool_size], remove_count, replace=False)

        removed_set = set(int(item) for item in removed)
        remaining = np.asarray(
            [item for item in sequence if int(item) not in removed_set],
            dtype=int,
        )
        return remaining, np.asarray(removed, dtype=int)

    def _repair(
        self,
        remaining: np.ndarray,
        removed: np.ndarray,
        operator_index: int,
    ) -> np.ndarray:
        sequence = remaining.tolist()
        if operator_index == 0:
            removed_order = removed.copy()
            self.rng.shuffle(removed_order)
            for task_index in removed_order:
                position = int(self.rng.integers(0, len(sequence) + 1))
                sequence.insert(position, int(task_index))
            return np.asarray(sequence, dtype=int)

        ranks = self._repair_ranks[operator_index]
        removed_order = removed[np.argsort(ranks[removed])]
        denominator = max(1, self.decoder.task_number - 1)
        for task_index in removed_order:
            relative_rank = float(ranks[task_index]) / denominator
            position = int(round(relative_rank * len(sequence)))
            sequence.insert(position, int(task_index))
        return np.asarray(sequence, dtype=int)

    def _update_operator_weights(self) -> None:
        used = self.operator_calls > 0
        performance = np.zeros_like(self.operator_weights)
        performance[used] = self.operator_scores[used] / self.operator_calls[used]
        if performance.sum() > 0:
            target = performance / performance.sum()
            self.operator_weights = (
                (1 - self.reaction_factor) * self.operator_weights
                + self.reaction_factor * target
            )
            self.operator_weights /= self.operator_weights.sum()
        self.operator_scores.fill(0)
        self.operator_calls.fill(0)

    def run(self) -> "CentralizedALNS":
        """Stage two: improve the centralized sequence with adaptive ALNS."""
        start = time.perf_counter()
        current_sequence = self._construct_initial_sequence()
        current_schedule = self.decoder.decode(current_sequence)
        best_sequence = current_sequence.copy()
        best_schedule = current_schedule
        temperature = self.initial_temperature
        non_improving = 0

        for iteration in range(1, self.max_iterations + 1):
            destroy_index, repair_index = self._select_operator_pair()
            remaining, removed = self._destroy(current_sequence, destroy_index)
            candidate_sequence = self._repair(remaining, removed, repair_index)
            candidate_schedule = self.decoder.decode(candidate_sequence)

            delta = candidate_schedule.benefit - current_schedule.benefit
            accepted = delta >= 0
            if not accepted and temperature > 0:
                probability = math.exp(max(-700.0, delta / temperature))
                accepted = bool(self.rng.random() < probability)

            if candidate_schedule.benefit > best_schedule.benefit + 1e-12:
                best_sequence = candidate_sequence.copy()
                best_schedule = candidate_schedule
                reward = 4.0
                non_improving = 0
            else:
                non_improving += 1
                reward = 2.0 if delta > 0 else (1.0 if accepted else 0.5)

            if accepted:
                current_sequence = candidate_sequence
                current_schedule = candidate_schedule

            self.operator_calls[destroy_index, repair_index] += 1
            self.operator_scores[destroy_index, repair_index] += reward
            if iteration % self.update_interval == 0:
                self._update_operator_weights()

            self.benefit_curve.append(best_schedule.benefit)
            temperature = max(self.minimum_temperature, temperature * self.cooling_rate)
            if non_improving >= self.max_non_improving:
                break

        self.time = time.perf_counter() - start
        self.best_sequence = best_sequence
        self.best_schedule = best_schedule
        self.decoder.validate(best_schedule)
        return self


def execute_c_alns(
    disposal_rate: float = 0.24,
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    max_iterations: int = 100,
    max_non_improving: int = 100,
    task_number_override: int | None = None,
) -> tuple[float, float, float]:
    """Run C-ALNS and return runtime, scheduled benefit, and total benefit."""
    algorithm = CentralizedALNS(
        config_path=config_path,
        max_iterations=max_iterations,
        max_non_improving=max_non_improving,
        disposal_rate=disposal_rate,
        task_number_override=task_number_override,
    ).run()
    assert algorithm.best_schedule is not None
    return (
        algorithm.time,
        algorithm.best_schedule.benefit,
        algorithm.decoder.total_benefit,
    )


if __name__ == "__main__":
    runtime, benefit, total = execute_c_alns()
    print(f"C-ALNS runtime: {runtime:.6f} s")
    print(f"C-ALNS BCR: {benefit / total:.6f}")
