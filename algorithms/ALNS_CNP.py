#!/usr/bin/env python3
import os

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
import random
import math
import copy
import pickle
from lib.progress import trange
import time

if scienceplots is not None:
    plt.style.use(['science', 'ieee'])

class ALNS(object):
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

    # operator manager
    destroy_list: list  # 2D list
    repair_list: list  # 2D list
    operator_score: np.ndarray
    operator_weight: np.ndarray
    operator_call_time: np.ndarray

    # iteration curve
    benefit_curve: list

    # status
    status: Status

    # performance index
    time: float

    # ramdom generator
    rng: np.random.default_rng

    def __init__(self, config_dir, max_iterations, max_not_improve, temperature, time_segments_iter, temperature_alpha, sigma1, sigma2,
                 sigma3, sigma4, rho, beta, gama):
        # world info
        self.config_dir = config_dir
        # algorithm parameters
        self.max_iterations = max_iterations  # 最大迭代次数
        self.max_not_Improve = max_not_improve  # 最大最优解不提升次数
        self.temperature = temperature  # 初始温度
        self.time_segments_iter = time_segments_iter  # 更新权重频率
        self.temperature_alpha = temperature_alpha  # 降温系数
        self.sigma1 = sigma1  # 提升最优解奖励
        self.sigma2 = sigma2  # 提升当前解奖励
        self.sigma3 = sigma3  # 更新当前解奖励
        self.sigma4 = sigma4  # 惩罚得分
        self.rho = rho  # 重新计算算子权重系数
        self.beta = beta  # 得分衰减系数
        self.gama = gama  # 任务删除百分比

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

        # 初始化状态
        self.status = Status()
        self.benefit_curve = []
        self.initial_operator_manager()

        # 随机生成器
        algorithm_seed = int(os.environ.get("SATELLITE_ALGORITHM_SEED", "8787"))
        self.rng = np.random.default_rng(algorithm_seed)

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

                # 3--5. Intersect service and visibility windows and build
                # bid packets only for DRSs that can see this task.
                parameter_package = build_parameter_packages(
                    index_iter,
                    self.tasklist,
                    self.visibility_windows,
                    self.satellite_list,
                    potential_conflict_list,
                    self.time_list,
                    self.bandwidth,
                )
                # 6. 计算可用时间窗<进程池><向卫星发出招标请求>
                futures = {executor.submit(self.find_time_slices, packet)
                           for packet in iter(parameter_package)
                           }
                concurrent.futures.wait(futures)
                # 7. 裁决任务<根据反馈的投标结果，裁标>
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

    def initial_operator_manager(self):
        """
        创建算子管理器
        :return: None
        """
        self.destroy_list = [self.ramdom_removal, self.min_priority_removal, self.max_conflict_removal,
                             self.min_efficiency_removal]
        self.repair_list = [self.random_insert, self.max_priority_insert, self.min_conflict_insert,
                            self.max_efficiency_insert]
        self.operator_score = np.ones((len(self.destroy_list), len(self.repair_list)))

        self.operator_weight = np.full((len(self.destroy_list), len(self.repair_list)), 1 / 16)
        self.operator_call_time = np.ones((len(self.destroy_list), len(self.repair_list)))

    def update_score(self, destroy_idx, repair_idx):
        """
        更新算子得分
        :return: None
        """
        # 1. 如果当前解获得新的全局最优解
        if self.status.NewBestSolution:
            self.operator_score[destroy_idx, repair_idx] += self.sigma1
            self.status.NewBestSolution = False

        # 2. 如果新解改善了当前解:
        elif self.status.ImproveCurrentSolution:
            self.operator_score[destroy_idx, repair_idx] += self.sigma2
            self.status.ImproveCurrentSolution = False

        # 3. 如果新解被接受了
        elif self.status.AcceptedAsCurrentSolution:
            self.operator_score[destroy_idx, repair_idx] += self.sigma3
            self.status.AcceptedAsCurrentSolution = False

        else:
            self.operator_score[destroy_idx, repair_idx] += self.sigma4

    def update_weight(self):
        """
        更新算子权重
        :return: None
        """
        self.status.NIterationRecomputeWeights = 0
        average_score: np.ndarray = self.operator_score / self.operator_call_time
        self.operator_weight = (1 - self.rho) * self.operator_weight + self.rho * average_score / average_score.sum()

        # decay the history experience & recover the time
        self.operator_score *= self.beta
        self.operator_call_time = np.ones((len(self.destroy_list), len(self.repair_list)))

    def update_call_times(self, destroy_idx, repair_idx):
        """
        更新算子调用次数
        :return: None
        """
        self.operator_call_time[destroy_idx, repair_idx] += 1

    def select_operator_pair(self):
        """Select one tendering-disposal combination from the 4 x 4 matrix."""
        probabilities = self.operator_weight.astype(float).ravel()
        probabilities /= probabilities.sum()
        pair_index = int(self.rng.choice(probabilities.size, p=probabilities))
        destroy_idx, repair_idx = np.unravel_index(pair_index, self.operator_weight.shape)
        return (
            self.destroy_list[destroy_idx],
            int(destroy_idx),
            self.repair_list[repair_idx],
            int(repair_idx),
        )

    def ramdom_removal(self):
        """
        随机删除
        :return: None
        """
        for satellite in self.satellite_list:

            # 1.calculate the task numbers
            num_task = math.ceil(len(satellite.execution_list) * self.gama)
            # executed_task = list(satellite.execution_list)
            # 2.randomly select the tasks
            task_to_remove = set(random.sample(list(satellite.execution_list), num_task))
            # 3. erase the record: time, score, frequency
            self.time_list[satellite.satellite_id] = [-1 if i in task_to_remove else num for i, num in
                                                      enumerate(self.time_list[satellite.satellite_id])]
            self.score_list[satellite.satellite_id] = [-1 if i in task_to_remove else num for i, num in
                                                       enumerate(self.score_list[satellite.satellite_id])]
            self.frequency_list[satellite.satellite_id] = [-1 if i in task_to_remove else num for i, num in
                                                           enumerate(self.frequency_list[satellite.satellite_id])]
            # 4. delete from the execution list
            satellite.update_execution_list(task_to_remove)

    def random_insert(self):
        """
        随机修复算子
        :return: None
        """
        executed_task = set.union(*[satellite.execution_list for satellite in self.satellite_list])
        not_executed_task = self.task_indices - executed_task
        # 1. transform set to list
        task_to_arrange = list(not_executed_task)
        # 2. ramdom shuffle the list
        random.shuffle(task_to_arrange)
        # 3. assign the tasks
        self.assign_tasks(task_to_arrange)

    def min_priority_removal(self):
        """
        最小优先级删除
        :return: None
        """
        for satellite in self.satellite_list:
            # 1. transform the task index to ndarray
            index_array = np.array(list(satellite.execution_list))
            # 2. sort the index by the priority
            sorted_indices = index_array[np.argsort(self.tasklist[index_array, :][:, 4])]
            # 3. select the top 10%
            task_to_remove = set(sorted_indices[:int(len(sorted_indices) * self.gama)])
            # 4. erase the record: time, score, frequency
            self.time_list[satellite.satellite_id] = [-1 if i in task_to_remove else num for i, num in
                                                      enumerate(self.time_list[satellite.satellite_id])]
            self.score_list[satellite.satellite_id] = [-1 if i in task_to_remove else num for i, num in
                                                       enumerate(self.score_list[satellite.satellite_id])]
            self.frequency_list[satellite.satellite_id] = [-1 if i in task_to_remove else num for i, num in
                                                           enumerate(self.frequency_list[satellite.satellite_id])]
            # 5. delete from the execution list
            satellite.update_execution_list(task_to_remove)

    def max_priority_insert(self):
        """
        最大优先级插入
        :return: None
        """
        executed_task = set.union(*[satellite.execution_list for satellite in self.satellite_list])
        not_executed_task = np.array(list(self.task_indices - executed_task))
        # 1. sort the not executed task by its priority
        task_to_arrange = np.array(not_executed_task)[np.argsort(self.tasklist[not_executed_task, 4])[::-1]]
        # 2. assign the tasks
        self.assign_tasks(task_to_arrange)

    def max_conflict_removal(self):
        """
        最大冲突删除
        :return: None
        """
        for satellite in self.satellite_list:
            # transform the task index to list
            index_array = list(satellite.execution_list)
            # 1.construct a tuple list: (task_index, conflict_num)
            index_count_pairs = [(i, len(self.conflict_list[i])) for i in index_array]
            # 2.sort the list
            sorted_pairs = sorted(index_count_pairs, key=lambda x: x[1], reverse=True)
            # 3.get the index in the sorted list
            sorted_indices = [pair[0] for pair in sorted_pairs]
            # 4. select the top 10%
            task_to_remove = set(sorted_indices[:int(len(sorted_indices) * self.gama)])
            # 5. erase the record: time, score, frequency
            self.time_list[satellite.satellite_id] = [-1 if i in task_to_remove else num for i, num in
                                                      enumerate(self.time_list[satellite.satellite_id])]
            self.score_list[satellite.satellite_id] = [-1 if i in task_to_remove else num for i, num in
                                                       enumerate(self.score_list[satellite.satellite_id])]
            self.frequency_list[satellite.satellite_id] = [-1 if i in task_to_remove else num for i, num in
                                                           enumerate(self.frequency_list[satellite.satellite_id])]
            # 6. delete from the execution list
            satellite.update_execution_list(task_to_remove)

    def min_conflict_insert(self):
        """
        最小冲突插入
        :return: None
        """
        executed_task = set.union(*[satellite.execution_list for satellite in self.satellite_list])
        # 1. transform the task index to ndarray
        not_executed_task = list(self.task_indices - executed_task)
        # 2. construct a tuple list: (task_index, conflict_num)
        index_count_pairs = [(i, len(self.conflict_list[i])) for i in not_executed_task]
        # 3. sort the tuple list
        sorted_pairs = sorted(index_count_pairs, key=lambda x: x[1])
        # 4. get the index in the sorted tuple list
        task_to_arrange = [pair[0] for pair in sorted_pairs]
        # 5. assign the tasks
        self.assign_tasks(task_to_arrange)

    def min_efficiency_removal(self):
        """
        最低效率删除
        :return: None
        """
        for satellite in self.satellite_list:
            # 1. transform the task index to ndarray
            index_array = np.array(list(satellite.execution_list))
            # 2. sort the index by the efficiency
            sorted_indices = index_array[np.argsort(self.tasklist[index_array, :][:, 5])]
            # 3. select the top 10%
            task_to_remove = set(sorted_indices[:int(len(sorted_indices) * self.gama)])
            # 4. erase the record: time, score, frequency
            self.time_list[satellite.satellite_id] = [-1 if i in task_to_remove else num for i, num in
                                                      enumerate(self.time_list[satellite.satellite_id])]
            self.score_list[satellite.satellite_id] = [-1 if i in task_to_remove else num for i, num in
                                                       enumerate(self.score_list[satellite.satellite_id])]
            self.frequency_list[satellite.satellite_id] = [-1 if i in task_to_remove else num for i, num in
                                                           enumerate(self.frequency_list[satellite.satellite_id])]
            # 5. delete from the execution list
            satellite.update_execution_list(task_to_remove)

    def max_efficiency_insert(self):
        """
        最大效率插入算子
        :return: None
        """
        executed_task = set.union(*[satellite.execution_list for satellite in self.satellite_list])
        not_executed_task = np.array(list(self.task_indices - executed_task))
        # 1. sort the not executed task by its efficiency
        task_to_arrange = np.array(not_executed_task)[np.argsort(self.tasklist[not_executed_task, 5])[::-1]]
        # 2. assign the tasks
        self.assign_tasks(task_to_arrange)

    def evaluate(self):
        """
        评价函数
        :return: fitness
        """
        score_list = np.array(self.score_list)

        return np.sum(score_list[score_list != -1])

    def backup_solution(self):
        """
        备份解
        :return: backup_scheme
        """
        backup_satellite_list = copy.deepcopy(self.satellite_list)
        backup_time_list = copy.deepcopy(self.time_list)
        backup_score_list = copy.deepcopy(self.score_list)
        backup_frequency_list = copy.deepcopy(self.frequency_list)

        return [backup_satellite_list, backup_time_list, backup_score_list, backup_frequency_list]

    def dump_global_best_solution(self):
        """
        将最优解dump成pickle文件
        :return: None
        """
        solution_package = [self.satellite_list, self.time_list, self.score_list, self.frequency_list]

        with open('../data/best_solution_alns.pkl', 'wb') as f:
            pickle.dump(solution_package, f)

    @staticmethod
    def load_global_best_solution():
        """
        加载全局最优解
        :return: global_best_solution
        """
        # 加载数据从文件
        with open('../data/best_solution_alns.pkl', 'rb') as f:
            solution_package = pickle.load(f)
        return solution_package

    def run(self):
        """
        自适应大规模邻域搜索算法执行入口
        :return: None
        """
        # 初始化任务
        self.initial_task_world()
        # 生成初始解
        start = time.time()
        self.construct_initial_solution()
        # 存储全局最优解
        global_best_benefit = self.evaluate()
        current_best_benefit = global_best_benefit
        self.dump_global_best_solution()
        # 记录迭代曲线
        self.benefit_curve.append(global_best_benefit)

        for _ in trange(self.max_iterations, desc="Iteration epoch"):
            # 1. backup current best data & update status
            self.status.NIterationRecomputeWeights += 1
            self.status.NIterationWithoutImprovement += 1
            current_best_solution = self.backup_solution()

            # 2. select operator
            destroy_method, destroy_index, repair_method, repair_index = self.select_operator_pair()

            # 3. execute the operator
            destroy_method()
            repair_method()

            # 4. evaluate
            new_solution_benefit = self.evaluate()

            # 5. Metropolis recept
            # 5.1 achive global best
            if new_solution_benefit > global_best_benefit:
                # 5.1.1 update status
                self.status.NewBestSolution = True
                self.status.NIterationWithoutImprovement = 0

                # 5.1.2 process global data
                global_best_benefit = new_solution_benefit
                self.dump_global_best_solution()

                # 5.1.3 process current data
                current_best_benefit = new_solution_benefit

                # 5.1.4 record the iteration curve
                self.benefit_curve.append(global_best_benefit)

            # 5.2 achieve new best
            elif new_solution_benefit > current_best_benefit:
                # 5.2.1 update status
                self.status.ImproveCurrentSolution = True

                # 5.2.2 process current data
                current_best_benefit = new_solution_benefit

                # 5.2.3 record the iteration curve
                self.benefit_curve.append(global_best_benefit)

            # 5.3 metropolis recept new data
            elif np.random.uniform() < math.exp((new_solution_benefit - current_best_benefit) / self.temperature):
                # 5.3.1 update status
                self.status.AcceptedAsCurrentSolution = True

                # 5.3.2 process current data
                current_best_benefit = new_solution_benefit

                # 5.3.3 record the iteration curve
                self.benefit_curve.append(global_best_benefit)

            # 5.4 fallback to the previous data
            else:
                # 5.4.1 fallback the data to previous
                self.satellite_list, self.time_list, self.score_list, self.frequency_list = current_best_solution

                # 5.4.2 record the iteration curve
                self.benefit_curve.append(global_best_benefit)

            # 6. Cooling the fire
            self.temperature = max(0.01, self.temperature * self.temperature_alpha)

            # 7. manage the operator
            self.update_call_times(destroy_index, repair_index)
            self.update_score(destroy_index, repair_index)

            # 8. update the operator weight
            if self.status.NIterationRecomputeWeights == self.time_segments_iter:
                self.update_weight()

            # 9. stop criterion
            if self.status.NIterationWithoutImprovement == self.max_not_Improve:
                break

        end = time.time()
        self.time = end - start
        print(f"Runtime: {self.time:.6f} s")

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
        # fig_schedule.legend(handles, labels, bbox_to_anchor=(1, 1), loc='upper left', framealpha=1)
        plt.savefig("spectrum.pdf")
        plt.show()

    def plot_curve(self):
        """
        绘制迭代收益曲线
        :return: None
        """
        fig_convergence, ax = plt.subplots()
        # Create the plot
        iterations = range(len(self.benefit_curve))
        ax.plot(iterations, self.benefit_curve, color="red", marker='*', markersize=4)

        # Add labels and title
        ax.set_xlabel('Iterations', fontsize=8,  labelpad=2)
        ax.set_ylabel('Fitness', fontsize=8,  labelpad=2)
        ax.set_title('Algorithm Convergence Curve', fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

        # Customize the plot
        plt.grid(False)
        # plt.ylim(0, 1)

        # Show the plot
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

    def capture_weight(self):
        """
        store the weight
        :return:
        """
        return self.operator_weight

def execute_alns_cnp(gama, _max_not_improve=5, _time_segments_iter=1):
    """
    execute function
    :return:
    """
    config_file_name = "../data/configure.json"
    _max_iterations = 100
    _temperature = 100
    _temperature_alpha = 0.95
    _sigma1 = 4
    _sigma2 = 2
    _sigma3 = 1
    _sigma4 = 0.5
    _rho = 0.2
    _beta = 0.2
    _gama = gama
    algorithm = ALNS(config_dir=config_file_name, max_iterations=_max_iterations, max_not_improve=_max_not_improve,
                     temperature=_temperature, time_segments_iter=_time_segments_iter,
                     temperature_alpha=_temperature_alpha,
                     sigma1=_sigma1, sigma2=_sigma2, sigma3=_sigma3, sigma4=_sigma4, rho=_rho, beta=_beta, gama=_gama)
    algorithm.run()
    if _max_not_improve == 5:
        return algorithm.time, algorithm.benefit_curve[-1], np.sum(algorithm.tasklist[:, 4])
    else:
        return algorithm.capture_weight()


if __name__ == "__main__":
    Config_file_name = "../data/configure.json"
    Max_iterations = 100
    Max_not_improve = 5
    Temperature = 100
    Time_segments_iter = 1
    Temperature_alpha = 0.95
    Sigma1 = 4
    Sigma2 = 2
    Sigma3 = 1
    Sigma4 = 0.5
    Rho = 0.2
    Beta = 0.2
    Gama = 0.2
    Algorithm = ALNS(config_dir=Config_file_name, max_iterations=Max_iterations, max_not_improve=Max_not_improve,
                     temperature=Temperature, time_segments_iter=Time_segments_iter, temperature_alpha=Temperature_alpha,
                     sigma1=Sigma1, sigma2=Sigma2, sigma3=Sigma3, sigma4=Sigma4, rho=Rho, beta=Beta, gama=Gama)
    Algorithm.run()
    # plot schedule
    Algorithm.plot_schedule()
    # plot iteration curve
    Algorithm.plot_curve()
