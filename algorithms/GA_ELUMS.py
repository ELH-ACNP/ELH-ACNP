#!/usr/bin/env python3
from __future__ import annotations

import math
import os
from pathlib import Path
import time

import numpy as np

from algorithms.C_ALNS import CentralizedDecoder, CentralizedSchedule, DEFAULT_CONFIG


class GAELUMS:
    """Population-based centralized search using DRS task permutations."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG,
        population_size: int = 50,
        max_generations: int = 100,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.3,
        tournament_size: int = 5,
        greedy_mutation_points: int = 5,
        reaction_factor: float = 0.2,
        task_number_override: int | None = None,
    ) -> None:
        if population_size < 4:
            raise ValueError("population_size must be at least four")
        if max_generations < 1:
            raise ValueError("max_generations must be positive")
        if not 0 <= mutation_rate <= 1 or not 0 <= crossover_rate <= 1:
            raise ValueError("Genetic probabilities must be in [0, 1]")

        self.decoder = CentralizedDecoder(config_path, task_number_override)
        self.population_size = int(population_size)
        self.max_generations = int(max_generations)
        self.mutation_rate = float(mutation_rate)
        self.crossover_rate = float(crossover_rate)
        self.tournament_size = min(int(tournament_size), self.population_size)
        self.greedy_mutation_points = max(2, int(greedy_mutation_points))
        self.reaction_factor = float(reaction_factor)
        self.rng = np.random.default_rng(
            int(os.environ.get("SATELLITE_ALGORITHM_SEED", "3845"))
        )

        self.mutation_weights = np.full(4, 0.25, dtype=float)
        self.mutation_scores = np.zeros(4, dtype=float)
        self.mutation_calls = np.zeros(4, dtype=int)
        self.selection_weights = np.full(2, 0.5, dtype=float)
        self.selection_scores = np.zeros(2, dtype=float)
        self.selection_calls = np.zeros(2, dtype=int)

        self.best_sequence: np.ndarray | None = None
        self.best_schedule: CentralizedSchedule | None = None
        self.benefit_curve: list[float] = []
        self.time = 0.0
        self._fitness_cache: dict[bytes, float] = {}

        self.conflict_degree = np.asarray(
            [len(item) for item in self.decoder.conflict_sets], dtype=float
        )
        self.conflicting_request_mask = self.conflict_degree > 0
        self.greedy_score = self._build_greedy_score()

    @staticmethod
    def _normalise(values: np.ndarray) -> np.ndarray:
        span = float(np.max(values) - np.min(values))
        if span <= 0:
            return np.zeros_like(values, dtype=float)
        return (values - np.min(values)) / span

    def _build_greedy_score(self) -> np.ndarray:
        tasklist = self.decoder.tasklist
        benefit = self._normalise(tasklist[:, 4])
        efficiency = self._normalise(tasklist[:, 5])
        conflict = self._normalise(self.conflict_degree)
        slack = (tasklist[:, 1] - tasklist[:, 0]) - tasklist[:, 3]
        urgency = 1.0 - self._normalise(slack)
        return 0.4 * benefit + 0.3 * efficiency + 0.2 * urgency + 0.1 * (1 - conflict)

    def _fitness(self, sequence: np.ndarray) -> float:
        key = sequence.tobytes()
        if key not in self._fitness_cache:
            self._fitness_cache[key] = self.decoder.decode(sequence).benefit
        return self._fitness_cache[key]

    def _initial_population(self) -> list[np.ndarray]:
        """Generate a diverse, reproducible population over the task order."""
        tasklist = self.decoder.tasklist
        bases = [
            np.argsort(-self.greedy_score),
            np.argsort(-tasklist[:, 4]),
            np.argsort(-tasklist[:, 5]),
            np.lexsort((np.arange(self.decoder.task_number), tasklist[:, 1])),
        ]

        population: list[np.ndarray] = []
        seen: set[bytes] = set()

        def add(individual: np.ndarray) -> None:
            candidate = np.asarray(individual, dtype=int)
            key = candidate.tobytes()
            if key not in seen:
                seen.add(key)
                population.append(candidate.copy())

        for base in bases:
            add(base)
            if len(population) >= self.population_size:
                return population

        base = bases[0]
        design_index = 0
        maximum_attempts = 20 * self.population_size
        while len(population) < self.population_size and design_index < maximum_attempts:
            offset = int(
                math.floor(
                    design_index * self.decoder.task_number / self.population_size
                )
            )
            stride = 1 + (2 * design_index + 1) % max(1, self.decoder.task_number)
            while math.gcd(stride, self.decoder.task_number) != 1:
                stride += 1
            indices = (
                offset + stride * np.arange(self.decoder.task_number)
            ) % self.decoder.task_number
            add(base[indices])
            design_index += 1

        while len(population) < self.population_size:
            add(self.rng.permutation(self.decoder.task_number))
        return population

    def _select_parent(self, fitness: np.ndarray, method: int) -> int:
        if method == 0:
            candidates = self.rng.choice(
                len(fitness), self.tournament_size, replace=False
            )
            return int(candidates[np.argmax(fitness[candidates])])

        order = np.argsort(fitness)
        ranks = np.empty(len(fitness), dtype=float)
        ranks[order] = np.arange(1, len(fitness) + 1, dtype=float)
        probabilities = ranks / ranks.sum()
        return int(self.rng.choice(len(fitness), p=probabilities))

    def _ordered_crossover(
        self,
        parent_a: np.ndarray,
        parent_b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        task_number = self.decoder.task_number
        start, end = np.sort(self.rng.choice(task_number, 2, replace=False))

        def make_child(first: np.ndarray, second: np.ndarray) -> np.ndarray:
            child = np.full(task_number, -1, dtype=int)
            child[start : end + 1] = first[start : end + 1]
            retained = set(int(item) for item in child[start : end + 1])
            insertion_positions = list(range(end + 1, task_number)) + list(
                range(0, start)
            )
            source = np.concatenate((second[end + 1 :], second[: end + 1]))
            genes = [int(item) for item in source if int(item) not in retained]
            child[insertion_positions] = genes
            return child

        return make_child(parent_a, parent_b), make_child(parent_b, parent_a)

    def _conflicting_positions(self, sequence: np.ndarray, count: int) -> np.ndarray:
        candidates = np.flatnonzero(self.conflicting_request_mask[sequence])
        if candidates.size < count:
            candidates = np.arange(sequence.size)
        return np.sort(self.rng.choice(candidates, count, replace=False))

    def _swap_mutation(self, sequence: np.ndarray) -> np.ndarray:
        child = sequence.copy()
        first, second = self._conflicting_positions(child, 2)
        child[first], child[second] = child[second], child[first]
        return child

    def _insertion_mutation(self, sequence: np.ndarray) -> np.ndarray:
        child = sequence.tolist()
        first, second = self._conflicting_positions(sequence, 2)
        gene = child.pop(int(first))
        child.insert(int(second), gene)
        return np.asarray(child, dtype=int)

    def _inversion_mutation(self, sequence: np.ndarray) -> np.ndarray:
        child = sequence.copy()
        start, end = self._conflicting_positions(child, 2)
        child[start : end + 1] = child[start : end + 1][::-1]
        return child

    def _greedy_multipoint_mutation(
        self,
        sequence: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        point_count = min(self.greedy_mutation_points, sequence.size)
        positions = self._conflicting_positions(sequence, point_count)
        genes = sequence[positions]
        tasklist = self.decoder.tasklist
        candidate_gene_orders = [
            genes[np.argsort(-self.greedy_score[genes])],
            genes[np.argsort(-tasklist[genes, 4])],
            genes[np.argsort(-tasklist[genes, 5])],
        ]

        best_child = sequence.copy()
        best_fitness = self._fitness(best_child)
        for gene_order in candidate_gene_orders:
            candidate = sequence.copy()
            candidate[positions] = gene_order
            candidate_fitness = self._fitness(candidate)
            if candidate_fitness > best_fitness + 1e-12:
                best_child = candidate
                best_fitness = candidate_fitness
        return best_child, best_fitness

    def _mutate(
        self,
        sequence: np.ndarray,
        operator_index: int,
    ) -> tuple[np.ndarray, float | None]:
        if operator_index == 0:
            return self._swap_mutation(sequence), None
        if operator_index == 1:
            return self._insertion_mutation(sequence), None
        if operator_index == 2:
            return self._inversion_mutation(sequence), None
        return self._greedy_multipoint_mutation(sequence)

    def _update_adaptive_weights(self) -> None:
        def update(
            weights: np.ndarray,
            scores: np.ndarray,
            calls: np.ndarray,
        ) -> None:
            performance = np.zeros_like(weights)
            used = calls > 0
            performance[used] = scores[used] / calls[used]
            if performance.sum() > 0:
                target = performance / performance.sum()
                weights[:] = (
                    (1 - self.reaction_factor) * weights
                    + self.reaction_factor * target
                )
                weights[:] = weights / weights.sum()
            scores.fill(0)
            calls.fill(0)

        update(self.mutation_weights, self.mutation_scores, self.mutation_calls)
        update(self.selection_weights, self.selection_scores, self.selection_calls)

    def run(self) -> "GAELUMS":
        start = time.perf_counter()
        population = self._initial_population()
        fitness = np.asarray([self._fitness(item) for item in population], dtype=float)

        best_index = int(np.argmax(fitness))
        best_sequence = population[best_index].copy()
        best_fitness = float(fitness[best_index])
        self.benefit_curve.append(best_fitness)

        for _ in range(self.max_generations):
            children: list[np.ndarray] = []
            child_fitness: list[float] = []

            while len(children) < self.population_size:
                selection_method = int(
                    self.rng.choice(2, p=self.selection_weights / self.selection_weights.sum())
                )
                first_parent = self._select_parent(fitness, selection_method)
                second_parent = self._select_parent(fitness, selection_method)
                while second_parent == first_parent:
                    second_parent = self._select_parent(fitness, selection_method)

                if self.rng.random() < self.crossover_rate:
                    offspring = self._ordered_crossover(
                        population[first_parent], population[second_parent]
                    )
                else:
                    offspring = (
                        population[first_parent].copy(),
                        population[second_parent].copy(),
                    )

                parent_reference = max(
                    float(fitness[first_parent]), float(fitness[second_parent])
                )
                family_best = -math.inf
                for child in offspring:
                    known_fitness: float | None = None
                    mutation_index: int | None = None
                    if self.rng.random() < self.mutation_rate:
                        probabilities = self.mutation_weights / self.mutation_weights.sum()
                        mutation_index = int(self.rng.choice(4, p=probabilities))
                        child, known_fitness = self._mutate(child, mutation_index)

                    value = (
                        float(known_fitness)
                        if known_fitness is not None
                        else self._fitness(child)
                    )
                    family_best = max(family_best, value)
                    children.append(child)
                    child_fitness.append(value)

                    if mutation_index is not None:
                        self.mutation_calls[mutation_index] += 1
                        self.mutation_scores[mutation_index] += (
                            4.0
                            if value > parent_reference + 1e-12
                            else (1.0 if value >= parent_reference - 1e-12 else 0.5)
                        )
                    if len(children) >= self.population_size:
                        break

                self.selection_calls[selection_method] += 1
                self.selection_scores[selection_method] += (
                    4.0
                    if family_best > parent_reference + 1e-12
                    else (1.0 if family_best >= parent_reference - 1e-12 else 0.5)
                )

            combined_population = population + children
            combined_fitness = np.concatenate(
                (fitness, np.asarray(child_fitness, dtype=float))
            )
            survivor_indices = np.argsort(-combined_fitness)[: self.population_size]
            population = [combined_population[int(index)] for index in survivor_indices]
            fitness = combined_fitness[survivor_indices]

            generation_best = int(np.argmax(fitness))
            if fitness[generation_best] > best_fitness + 1e-12:
                best_fitness = float(fitness[generation_best])
                best_sequence = population[generation_best].copy()
            self.benefit_curve.append(best_fitness)
            self._update_adaptive_weights()

        self.time = time.perf_counter() - start
        self.best_sequence = best_sequence
        self.best_schedule = self.decoder.decode(best_sequence)
        self.decoder.validate(self.best_schedule)
        return self


def execute_ga_elums(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    population_size: int = 50,
    max_generations: int = 100,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.3,
    tournament_size: int = 5,
    task_number_override: int | None = None,
) -> tuple[float, float, float]:
    """Run GA-ELUMS and return runtime, scheduled benefit, and total benefit."""
    algorithm = GAELUMS(
        config_path=config_path,
        population_size=population_size,
        max_generations=max_generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        tournament_size=tournament_size,
        task_number_override=task_number_override,
    ).run()
    assert algorithm.best_schedule is not None
    return (
        algorithm.time,
        algorithm.best_schedule.benefit,
        algorithm.decoder.total_benefit,
    )


if __name__ == "__main__":
    runtime, benefit, total = execute_ga_elums()
    print(f"GA-ELUMS runtime: {runtime:.6f} s")
    print(f"GA-ELUMS BCR: {benefit / total:.6f}")
