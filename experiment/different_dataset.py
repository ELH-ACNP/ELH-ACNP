#!/usr/bin/env python3
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(Path(__file__).resolve().parent)

from algorithms.ELH_ACNP import execute_pre_tuned_alns_cnp
import matplotlib.pyplot as plt
try:
    import scienceplots
except ImportError:
    scienceplots = None
import numpy as np
import json
import pickle

if scienceplots is not None:
    plt.style.use(['science', 'ieee'])

class Executor:
    """
    Experiment Repeater
    """
    runtimes: list
    benefits: list
    total_benefits: list

    def __init__(self, n, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.execute_times = n
        self.runtimes = []
        self.benefits = []
        self.total_benefits = []

    def run_function_n_times(self):
        """
        run experiment repeat n times
        :return:
        """
        task_seed = int(os.environ["SATELLITE_TASK_SEED"])
        for run_index in range(self.execute_times):
            algorithm_seed = 50_000 + run_index
            os.environ["SATELLITE_TASK_SEED"] = str(task_seed)
            os.environ["SATELLITE_ALGORITHM_SEED"] = str(algorithm_seed)
            random.seed(algorithm_seed)
            np.random.seed(algorithm_seed)
            runtime, benefit, total_benefit = self.func(*self.args, **self.kwargs)
            self.runtimes.append(runtime)
            self.benefits.append(benefit)
            self.total_benefits.append(total_benefit)

    def analyse_result(self):
        """
        analyse the result
        :return:
        """
        mean_run_time = np.mean(self.runtimes)

        mean_bid_time = np.mean(self.benefits)
        mean_disposal_time = np.mean(self.total_benefits)
        return mean_run_time, mean_bid_time, mean_disposal_time

    def reset_container(self):
        """
        reset result container
        :return:
        """
        self.runtimes = []
        self.benefits = []
        self.total_benefits = []

class JsonModifier:
    """
    Experiment Parameter Modifier
    """
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """
        load the json
        :return:
        """
        with open(self.file_path, 'r') as f:
            return json.load(f)

    def write_data(self, data):
        """
        update the information in json
        :param data:
        :return:
        """
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=4, sort_keys=True)

    def modify_and_save(self, updates_dict):
        """
        call this function to execute
        :param updates_dict:
        :return:
        """
        data = self.load_data()
        for key, new_value in updates_dict.items():
            data[key] = new_value
        self.write_data(data)

def run_and_analyse(executor):
    """
    MetaExecutor
    :param executor:
    :return:
    """
    executor.run_function_n_times()
    results = executor.analyse_result()
    executor.reset_container()

    return results


if __name__ != "__main__":
    pass
else:
    # set experiment parameter
    start, end, step = 50, 1501, 5
    seeds = [222, 444, 666, 888, 114514]
    repeat_times = 20
    gama = 0.24
    json_modifier = JsonModifier("../data/configure.json")
    json_modifier_2 = JsonModifier("../data/seed.json")

    executor_pre_tuned_alns_cnp = Executor(
        repeat_times,
        execute_pre_tuned_alns_cnp,
        gama,
        _return_components=True,
    )

    Result = {
        222: [],
        444: [],
        666: [],
        888: [],
        114514: []
    }

    # evaluation system: benefits  runtimes total_benefits
    # create repeater and execute it for appointed times
    for seed in seeds:
        os.environ["SATELLITE_TASK_SEED"] = str(seed)
        json_modifier_2.modify_and_save({
                "seed": seed
            }
            )
        for task_num in np.arange(start, end, step):
            json_modifier.modify_and_save({
                "Task_Number": int(task_num),
                "Satellite_Number": 5
            }
            )

            result_pre_tuned_alns_cnp = run_and_analyse(executor_pre_tuned_alns_cnp)
            Result[seed].append(result_pre_tuned_alns_cnp)

    with open('../data/result_seed_task.pkl', 'wb') as f:
        pickle.dump(Result, f)


