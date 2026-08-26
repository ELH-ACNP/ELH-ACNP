#!/usr/bin/env python3
import json
import time


class Helper(object):
    config_dir: str
    task_info: list
    resource_info: list
    scene_info: list

    def __init__(self, config_dir):
        self.config_dir = config_dir
        self.task_info = []
        self.resource_info = []
        self.scene_info = []

    def parse_world_info(self):
        """
        解析任务信息
        :return:
            task_info：    任务信息
            scene_info：   场景信息
            resource_info：资源信息
        """
        with open(self.config_dir, encoding="utf-8") as json_file:
            config_data = json.load(json_file)

        task_num = config_data["Task_Number"]
        satellite_num = config_data["Satellite_Number"]
        unit_bandwidth = config_data["Unit_Bandwidth"]
        max_frequency = config_data["Spectrum_Range"]["Maximum_Frequency"]
        min_frequency = config_data["Spectrum_Range"]["Minimum_Frequency"]
        start_time = config_data["Simulation_Period"]["Simulation_Start"]
        end_time = config_data["Simulation_Period"]["Simulation_End"]
        distribute_mean_1 = config_data["Task_Distribute"]["Distribute_Mean_1"]
        distribute_std_1 = config_data["Task_Distribute"]["Distribute_std_1"]
        distribute_mean_2 = config_data["Task_Distribute"]["Distribute_Mean_2"]
        distribute_std_2 = config_data["Task_Distribute"]["Distribute_std_2"]

        self.task_info = [task_num,
                          distribute_mean_1 / 86400,
                          distribute_std_1 / 86400,
                          distribute_mean_2 / 86400,
                          distribute_std_2 / 86400
                          ]
        start_timestamp = int(time.mktime(time.strptime(start_time, "%Y-%m-%d %H:%M:%S")))
        end_timestamp = int(time.mktime(time.strptime(end_time, "%Y-%m-%d %H:%M:%S")))
        scheduling_horizon = (end_timestamp - start_timestamp) / 86400

        # All temporal quantities are represented in days relative to the
        # beginning of the scheduling horizon.  This keeps the task-window and
        # execution-duration units consistent.
        self.scene_info = [0.0,
                           scheduling_horizon
                           ]

        self.resource_info = [satellite_num,
                              unit_bandwidth,
                              0,
                              max_frequency - min_frequency
                              ]

        return self.task_info, self.scene_info, self.resource_info


if __name__ == "__main__":
    configure_dir = "../data/configure.json"
    helper = Helper(configure_dir)
    print(helper.parse_world_info())
