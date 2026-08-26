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
    benefit_curves: list

    def __init__(self, n, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.execute_times = n
        self.benefit_curves = []

    def run_function_n_times(self):
        """
        run experiment repeat n times
        :return:
        """
        for run_index in range(self.execute_times):
            task_seed = 10_000 + run_index
            algorithm_seed = 50_000 + run_index
            os.environ["SATELLITE_TASK_SEED"] = str(task_seed)
            os.environ["SATELLITE_ALGORITHM_SEED"] = str(algorithm_seed)
            random.seed(algorithm_seed)
            np.random.seed(algorithm_seed)
            benefit_curve = self.func(*self.args, **self.kwargs)
            self.benefit_curves.append(benefit_curve)
        return self.benefit_curves

    def reset_container(self):
        """
        reset result container
        :return:
        """
        self.benefit_curves = []

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
    results = executor.run_function_n_times()
    executor_pre_tuned_alns_cnp.reset_container()

    return results[0]


if __name__ != "__main__":
    pass
else:
    # set experiment parameter
    start, end, step = 3, 7, 1
    repeat_times = 1
    gama = 0.24
    max_not_improve = 100
    json_modifier = JsonModifier("../data/configure.json")

    executor_pre_tuned_alns_cnp = Executor(
        repeat_times,
        execute_pre_tuned_alns_cnp,
        gama,
        max_not_improve,
        _return_bcr_curve=True,
    )

    Result = {
        3: [],
        4: [],
        5: [],
        6: [],
    }

    # evaluation system: benefits  runtimes total_benefits
    # create repeater and execute it for appointed times
    fig, ax = plt.subplots(dpi=200, figsize=(8, 6))
    # ax.set_xlim([50, 1500])
    # ax.set_ylim([0, 40])
    x_values = np.arange(0, 101, 1)
    for sat_num in np.arange(start, end, step):
        json_modifier.modify_and_save({
            "Task_Number": 1500,
            "Satellite_Number": int(sat_num)
        }
        )

        result_pre_tuned_alns_cnp = run_and_analyse(executor_pre_tuned_alns_cnp)
        Result[sat_num] = result_pre_tuned_alns_cnp

    with open('../data/result_convergence.pkl', 'wb') as f:
        pickle.dump(Result, f)

