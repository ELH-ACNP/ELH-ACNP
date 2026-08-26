#!/usr/bin/env python3
import os

import numpy as np
from task_tool.InitWrapper import initialize_attributes
from task_tool.Visibility import (
    generate_task_locations,
    generate_visibility_windows,
    load_visibility_settings,
)


@initialize_attributes
class Task(object):
    def __init__(self, task_number, start, end, mean, std, mean2, std2, config_dir=None):
        self.task_number = task_number
        self.tasklist = None
        self.conflict_set = None
        self.start = start
        self.end = end
        self.mean = mean
        self.std = std
        self.mean2 = mean2
        self.std2 = std2
        self.config_dir = config_dir
        self.task_locations = None
        self.satellite_longitudes = None
        self.elevation_angles = None
        self.visibility_matrix = None
        self.visibility_windows = None

    def task_generator(self):
        """
        按照一定分布生成任务
        :return: None
        """
        # A local generator makes task instances reproducible without resetting
        # the random stream used by the scheduling algorithms.
        task_seed = int(os.environ.get("SATELLITE_TASK_SEED", "666"))
        rng = np.random.RandomState(task_seed)
        band_choice = np.array(np.arange(3, 6))
        self.tasklist = np.zeros((self.task_number, 5))
        for i in range(self.task_number):
            while True:
                window_length = float(rng.normal(self.mean, self.std))
                if 0 < window_length <= self.end - self.start:
                    a = float(rng.uniform(self.start, self.end - window_length))
                    b = a + window_length
                    self.tasklist[i, 0] = a  # 任务开始时间
                    self.tasklist[i, 1] = b  # 任务结束时间
                    break
            self.tasklist[i, 2] = rng.choice(band_choice)  # 任务占用频带

            while True:
                task_len = float(rng.normal(self.mean2, self.std2))  # 任务时长
                if 0 < task_len <= self.tasklist[i, 1] - self.tasklist[i, 0]:
                    self.tasklist[i, 3] = task_len
                    break

            # self.tasklist[i, 4] = np.random.randint(1, self.sc + 1)  # 任务与卫星绑定
            self.tasklist[i, 4] = rng.uniform(0, 1) * 10  # 任务权重

        # add efficiency indicator
        efficiency = self.tasklist[:, 4] / (self.tasklist[:, 2] * self.tasklist[:, 3])
        self.tasklist = np.column_stack((self.tasklist, efficiency))

    def visibility_generator(self):
        """Generate task locations and satellite-specific visibility windows."""
        if self.config_dir is None:
            self.task_locations = np.empty((self.task_number, 2), dtype=float)
            self.satellite_longitudes = np.empty(0, dtype=float)
            self.elevation_angles = np.empty((self.task_number, 0), dtype=float)
            self.visibility_matrix = np.empty((self.task_number, 0), dtype=bool)
            self.visibility_windows = np.empty((self.task_number, 0, 2), dtype=float)
            return

        settings = load_visibility_settings(self.config_dir)
        task_seed = int(os.environ.get("SATELLITE_TASK_SEED", "666"))
        self.task_locations = generate_task_locations(
            self.task_number,
            task_seed,
            settings,
        )
        self.satellite_longitudes = settings["satellite_longitudes_deg"]
        (
            self.elevation_angles,
            self.visibility_matrix,
            self.visibility_windows,
        ) = generate_visibility_windows(
            self.task_locations,
            self.satellite_longitudes,
            float(settings["minimum_elevation_deg"]),
            self.start,
            self.end,
        )

    def task_conflict_calculate(self):
        """
        构建每个任务的冲突集
        :return: None
        """
        self.conflict_set = [[] for _ in range(self.task_number)]
        for index, row in enumerate(self.tasklist):
            for sub_index, sub_row in enumerate(self.tasklist[index + 1:], start=index + 1):
                start_time1 = row[0]
                end_time1 = row[1]
                start_time2 = sub_row[0]
                end_time2 = sub_row[1]
                if not(start_time1 > end_time2 or start_time2 > end_time1):
                    self.conflict_set[index].append(sub_index)
                    self.conflict_set[sub_index].append(index)
        # converse to set the inner element
        self.conflict_set = [set(tuple(element)) for element in self.conflict_set]

    def run(self):
        self.task_generator()
        self.visibility_generator()
        self.task_conflict_calculate()


if __name__ == "__main__":
    task = Task(600, 738188, 738188.5, 1/36, 1/288, 1/92, 1/720)
    print(task.tasklist[:, 5])
    print(task.conflict_set[1])
    # print(task.conflict_set)
