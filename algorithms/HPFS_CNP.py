#!/usr/bin/env python3
from task_tool.Task_generator import Task
from task_tool.Visibility import build_parameter_packages
from task_tool.HelpLibrary import Helper
from lib.Agent import Satellite
from lib.Status import Status
import concurrent.futures
import matplotlib.pyplot as plt
import matplotlib.patches as patches
try:
    import scienceplots
except ImportError:
    scienceplots = None
import numpy as np
import time

if scienceplots is not None:
    plt.style.use(['science', 'ieee'])

class HPFS(object):
    bandwidth: int
    task_num: int
    satellite_num: int
    tasklist: np.ndarray
    task_indices: set
    conflict_list: list
    satellite_list: list[Satellite]

    # main result
    score_list: list  # 2D list
    time_list: list  # 2D list
    frequency_list: list  # 2D list

    # performance index
    time: float
    benefit: float

    def __init__(self, config_dir):
        # world info
        self.config_dir = config_dir

    def initial_task_world(self):
        """
        初始化任务, 赋值类属性：task_num, satellite_num, bandwidth, conflict_set, tasklist
        task_info:[task_num,
                  distribute_mean_1,
                  distribute_std_1,
                  distribute_mean_2,
                  distribute_std_2
                 ]
        scene_info:[start_time,
                   end_time
                   ]
        resource_info:[satellite_num,
                      unit_bandwidth,
                      max_frequency,
                      min_frequency
                     ]
        :return: None
        """
        # 读取文件
        file_helper = Helper(self.config_dir)
        task_info, scene_info, resource_info = file_helper.parse_world_info()
        self.task_num, distribute_mean_1, distribute_std_1, distribute_mean_2, distribute_std_2 = task_info
        self.satellite_num, _, _, self.bandwidth = resource_info
        # 生成任务
        task = Task(self.task_num,
                    scene_info[0], scene_info[1],
                    distribute_mean_1, distribute_std_1,
                    distribute_mean_2, distribute_std_2,
                    config_dir=self.config_dir,
                    )
        # 生成卫星集
        self.satellite_list = [
            Satellite(self.bandwidth, i, task.satellite_longitudes[i])
            for i in range(self.satellite_num)
        ]
        self.tasklist = task.tasklist
        self.task_locations = task.task_locations
        self.visibility_matrix = task.visibility_matrix
        self.visibility_windows = task.visibility_windows
        self.conflict_list = task.conflict_set
        self.task_indices = set(range(self.task_num))

        # 初始化参数
        self.score_list = [[-1] * self.task_num for _ in range(self.satellite_num)]
        self.time_list = [[-1] * self.task_num for _ in range(self.satellite_num)]
        self.frequency_list = [[-1] * self.task_num for _ in range(self.satellite_num)]

    def assign_tasks(self, task_sequences):
        """
        根据输入序列分配任务
        :param task_sequences:
        :return: None
        """
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.satellite_num) as executor:
            for index_iter in iter(task_sequences):

                # 1. 找出任务冲突集
                potential_conflict_list = self.conflict_list[index_iter]

                # 2. 提取任务信息
                task_start_time, task_end_time, task_width, task_serve_time, task_priority, _ = self.tasklist[index_iter]

                # 3--5. Build bid packets for visible DRSs.
                parameter_package = build_parameter_packages(
                    index_iter,
                    self.tasklist,
                    self.visibility_windows,
                    self.satellite_list,
                    potential_conflict_list,
                    self.time_list,
                    self.bandwidth,
                )

                # 6. 计算可用时间窗<进程池>
                futures = {executor.submit(self.find_time_slices, packet)
                           for packet in iter(parameter_package)
                           }
                concurrent.futures.wait(futures)
                # 7. 裁决任务
                earliest_time = float('inf')
                best_frequency = None
                best_satellite = None
                for future in concurrent.futures.as_completed(futures):

                    if future.result() and future.result()[0] < earliest_time:
                        earliest_time = future.result()[0]
                        best_frequency = future.result()[1]
                        best_satellite = future.result()[2]

                if best_frequency:
                    # Record the selected DRS, time, benefit, and frequency slot.
                    self.satellite_list[best_satellite].add_task(index_iter)
                    self.time_list[best_satellite][index_iter] = earliest_time
                    self.score_list[best_satellite][index_iter] = self.tasklist[index_iter, 4]
                    self.frequency_list[best_satellite][index_iter] = self.bandwidth - best_frequency

    @staticmethod
    def find_time_slices(args):
        """
        寻找时间窗<找到第一个窗口直接返回，不找出全部的>
        :param args:
                -bandwidth:
                -requested_time_window:
                -slice_length:
                -slice_width:
                -occupied_time_windows:
        :return: resource information or None
        """
        # unpack the parameter package
        bandwidth, requested_time_window, slice_length, slice_width, occupied_time_windows, satellite_id = args

        if not occupied_time_windows:
            return [requested_time_window[0], bandwidth, satellite_id]

        start_time, end_time = requested_time_window

        events = [(start_time, 0), (end_time, 0)]
        # Add start and end events
        # Add occupied_time_window events
        for occupied_start, duration, occupied_width in occupied_time_windows:
            if occupied_start < end_time and occupied_start + duration > start_time:
                events.append((max(start_time, occupied_start), occupied_width))
                events.append((min(end_time, occupied_start + duration), -occupied_width))
        # Sort by time
        events.sort(key=lambda event: (event[0], -event[1]))

        curr_width = bandwidth
        min_width = bandwidth
        last_event_time = start_time
        for time, width_change in events:
            if curr_width >= slice_width and time - last_event_time > slice_length:
                # # Return the first available time window <accumulate version>
                # if curr_width - width_change < slice_width or time == end_time:
                return [last_event_time, min_width, satellite_id]
                # return [last_event_time, min_width]
                # simple version

            curr_width -= width_change
            min_width = min(min_width, curr_width)
            if curr_width < slice_width:
                last_event_time = np.nan
                min_width = curr_width

            elif np.isnan(last_event_time) and curr_width >= slice_width:
                last_event_time = time
                min_width = curr_width
        return None  # Return None if there is no available time window

    def construct_initial_solution(self):
        """
        构建自适应大规模邻域搜索算法初始解
        按照权重优先生成
        :return: None
        """
        sorted_index = np.argsort(self.tasklist[:, 4])[::-1]
        self.assign_tasks(sorted_index)

    def evaluate(self):
        """
        评价函数
        :return: fitness
        """
        score_list = np.array(self.score_list)

        return np.sum(score_list[score_list != -1])

    def run(self):
        """
        HPFS算法入口
        :return: None
        """
        # 初始化任务
        self.initial_task_world()
        # 生成初始解
        start = time.time()
        self.construct_initial_solution()
        end = time.time()
        self.time = end - start
        print(f"Runtime: {self.time:.6f} s")
        # evaluate
        self.benefit = float(self.evaluate())
        print("HPFS-CNP benefit:", self.benefit)

    def plot_schedule(self):
        """
        绘制资源占用图
        :return:
        """
        # plot agent schedules
        fig_schedule = plt.figure(1)
        fig_schedule.suptitle("Spectrum Resource Usage", fontsize=6, x=0.5, y=0.95, ha='center', va='top')
        fig_schedule.text(0.06, 0.5, 'Bandwidth', va='center', rotation='vertical', fontsize=6)
        for satellite in range(self.satellite_num):
            ax = plt.subplot(self.satellite_num, 1, satellite + 1)
            ax.set_title("Satellite " + str(satellite), fontsize=6, pad=3)
            ax.tick_params(axis='both', which='major', labelsize=6)
            if satellite == (self.satellite_num - 1):
                ax.set_xlabel("Time", fontsize=6)
            ax.set_xlim([0, 0.5])  # 时间范围
            ax.set_ylim([0, 22])

            # 设置颜色
            # colors = ['red', 'green', 'blue', 'orange', 'yellow', 'pink', 'purple', 'black', 'gray']
            events = self.converse_task_to_events(satellite)

            curr_width = 0
            last_event_time = 0
            for time, width_change in events:
                if time > last_event_time:
                    length = time - last_event_time
                    rect = patches.Rectangle((last_event_time, 0), time - last_event_time, curr_width,
                                             facecolor='#0096A6')
                    ax.add_patch(rect)
                    # 绘图
                last_event_time = time
                curr_width += width_change

        fig_schedule.subplots_adjust(hspace=0.5)

        # set legends
        colors = ["red", "red"]
        line_styles = ["-", "-."]
        line_width_list = [10, 2]
        labels = ["Assignment Time", "Task Time"]

        def f(line_style, color_type, line_width):
            return plt.plot([], [], linestyle=line_style, color=color_type,
                            linewidth=line_width)[0]

        handles = [f(line_styles[i], colors[i], line_width_list[i]) for i in range(len(labels))]
        fig_schedule.legend(handles, labels, bbox_to_anchor=(1, 1), loc='upper left', framealpha=1)

        plt.show()

    def converse_task_to_events(self, sat_index):
        """
        卫星任务集转事件集
        :param sat_index:
        :return: events
        """
        satellite = self.satellite_list[sat_index]
        occupied_windows = [[self.time_list[satellite.satellite_id][i],
                             self.tasklist[i, 3],
                             self.tasklist[i, 2]]
                            for i in satellite.execution_list
                            ]
        events = []
        # Add start and end events
        # Add occupied_time_window events
        for occupied_start, duration, occupied_width in occupied_windows:
            events.append((occupied_start, occupied_width))
            events.append((occupied_start + duration, -occupied_width))
        # Sort by time
        events.sort(key=lambda event: (event[0], -event[1]))
        return events

def execute_hpfs_cnp():
    """
    execute function
    :return:
    """
    config_file_name = "../data/configure.json"
    algorithm = HPFS(config_dir=config_file_name)
    algorithm.run()
    return algorithm.time, algorithm.benefit, np.sum(algorithm.tasklist[:, 4])


if __name__ == "__main__":
    Config_file_name = "../data/configure.json"
    Algorithm = HPFS(config_dir=Config_file_name)
    Algorithm.run()
    # plot schedule
    Algorithm.plot_schedule()

