#!/usr/bin/env python3
"""Fast validation checks for the scheduling implementation."""

from __future__ import annotations

import os
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.ELH_ACNP import ALNS
from algorithms.FCFS_CNP import FIFO
from algorithms.HPFS_CNP import HPFS
from lib.Agent import Satellite
from lib.Status import Status
from task_tool.HelpLibrary import Helper
from task_tool.Task_generator import Task
from task_tool.Visibility import build_parameter_packages


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.task_info, cls.scene_info, cls.resource_info = Helper(
            str(PROJECT_ROOT / "data" / "configure.json")
        ).parse_world_info()

    def make_task_instance(self, seed: int, count: int = 200, with_visibility: bool = True):
        os.environ["SATELLITE_TASK_SEED"] = str(seed)
        return Task(
            count,
            self.scene_info[0],
            self.scene_info[1],
            *self.task_info[1:],
            config_dir=(
                str(PROJECT_ROOT / "data" / "configure.json")
                if with_visibility
                else None
            ),
        )

    def make_tasks(self, seed: int, count: int = 200) -> np.ndarray:
        return self.make_task_instance(seed, count).tasklist

    def test_scene_units_and_bandwidth(self) -> None:
        self.assertEqual(self.scene_info, [0.0, 1.0])
        self.assertEqual(self.resource_info[-1], 20)

    def test_generated_tasks_are_valid_and_reproducible(self) -> None:
        first = self.make_tasks(666)
        repeated = self.make_tasks(666)
        different = self.make_tasks(667)
        np.testing.assert_allclose(first, repeated)
        self.assertFalse(np.array_equal(first, different))
        self.assertTrue(np.all(first[:, 0] >= 0))
        self.assertTrue(np.all(first[:, 1] <= 1))
        self.assertTrue(np.all(first[:, 3] > 0))
        self.assertTrue(np.all(first[:, 3] <= first[:, 1] - first[:, 0]))

    def test_visibility_geometry_and_independent_random_stream(self) -> None:
        task = self.make_task_instance(10000, 1500)
        no_visibility = self.make_task_instance(10000, 1500, with_visibility=False)
        np.testing.assert_allclose(task.tasklist, no_visibility.tasklist)
        np.testing.assert_allclose(
            task.satellite_longitudes,
            [76, 86, 96, 106, 116, 126],
        )
        self.assertEqual(task.task_locations.shape, (1500, 2))
        self.assertEqual(task.visibility_matrix.shape, (1500, 6))
        self.assertTrue(np.all(task.visibility_matrix))
        self.assertGreaterEqual(float(np.min(task.elevation_angles)), 5.0)
        self.assertTrue(np.all(task.task_locations[:, 0] >= 73.5))
        self.assertTrue(np.all(task.task_locations[:, 0] <= 135.0))
        self.assertTrue(np.all(task.task_locations[:, 1] >= 18.0))
        self.assertTrue(np.all(task.task_locations[:, 1] <= 53.5))

    def test_invisible_drs_is_excluded_from_bidding(self) -> None:
        tasklist = np.asarray([[0.1, 0.9, 3, 0.2, 5, 1]], dtype=float)
        satellites = [Satellite(20, 0, 76), Satellite(20, 1, 86)]
        visibility_windows = np.asarray([[[np.nan, np.nan], [0.2, 0.8]]])
        packages = build_parameter_packages(
            0,
            tasklist,
            visibility_windows,
            satellites,
            set(),
            [[-1], [-1]],
            20,
        )
        self.assertEqual(len(packages), 1)
        self.assertEqual(packages[0][1], [0.2, 0.8])
        self.assertEqual(packages[0][-1], 1)

    def test_hpfs_and_fcfs_ordering(self) -> None:
        tasklist = np.asarray([
            [0.7, 0.9, 3, 0.1, 5, 1],
            [0.1, 0.4, 3, 0.1, 1, 1],
            [0.4, 0.6, 3, 0.1, 9, 1],
        ], dtype=float)

        hpfs = HPFS.__new__(HPFS)
        hpfs.tasklist = tasklist
        hpfs_order: list[int] = []
        hpfs.assign_tasks = lambda sequence: hpfs_order.extend(int(i) for i in sequence)
        hpfs.construct_initial_solution()
        self.assertEqual(hpfs_order, [2, 0, 1])

        fcfs = FIFO.__new__(FIFO)
        fcfs.tasklist = tasklist
        fcfs_order: list[int] = []
        fcfs.assign_tasks = lambda sequence: fcfs_order.extend(int(i) for i in sequence)
        fcfs.construct_initial_solution()
        self.assertEqual(fcfs_order, [1, 2, 0])

    def test_scan_line_example(self) -> None:
        result = ALNS.find_time_slices([
            10,
            [0, 10],
            2,
            4,
            [[0, 3, 4], [0, 3, 3], [4, 3, 5]],
            0,
        ])
        self.assertEqual(result, [3, 5, 0])

    def test_adaptive_counter_and_joint_pair_selection(self) -> None:
        self.assertEqual(Status().NIterationRecomputeWeights, 0)
        algorithm = ALNS.__new__(ALNS)
        algorithm.operator_weight = np.zeros((4, 4), dtype=float)
        algorithm.operator_weight[3, 1] = 1
        algorithm.destroy_list = ["d0", "d1", "d2", "d3"]
        algorithm.repair_list = ["r0", "r1", "r2", "r3"]
        algorithm.rng = np.random.default_rng(1)
        self.assertEqual(algorithm.select_operator_pair(), ("d3", 3, "r1", 1))

    def test_pretrained_weight_matrix_is_available(self) -> None:
        with (PROJECT_ROOT / "data" / "weight_trained.pkl").open("rb") as stream:
            weight = np.asarray(pickle.load(stream), dtype=float)
        self.assertEqual(weight.shape, (4, 4))
        self.assertTrue(np.all(weight > 0))


if __name__ == "__main__":
    unittest.main()
