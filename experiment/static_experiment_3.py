#!/usr/bin/env python3
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(Path(__file__).resolve().parent)

from algorithms.ELH_ACNP import execute_pre_tuned_alns_cnp
from algorithms.ALNS_CNP import execute_alns_cnp
from algorithms.LNS_CNP import execute_lns_cnp
from algorithms.VND_CNP import execute_vnd_cnp
from algorithms.VNS_CNP import execute_vns_cnp
from algorithms.HCNP import execute_ns_cnp
from algorithms.FCFS_CNP import execute_fcfs_cnp
from algorithms.HPFS_CNP import execute_hpfs_cnp
from algorithms.LLF_CNP import execute_llf_cnp
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

    def modify_gama(self, *args, **kwargs):
        """
        modify the parameter gama
        :param args:
        :param kwargs:
        :return:
        """
        self.args = args
        self.kwargs = kwargs

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
        variance_run_time = np.std(self.runtimes, ddof=1)
        bcr = np.asarray(self.benefits) / np.asarray(self.total_benefits)
        return mean_run_time, variance_run_time, np.mean(bcr), np.std(bcr, ddof=1), 1.0

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

def modify_gama(_gama):
    """
    uniform modify the parameter gama
    :param _gama:
    :return:
    """
    executor_pre_tuned_alns_cnp.modify_gama(_gama)
    executor_alns_cnp.modify_gama(_gama)
    executor_lns_cnp.modify_gama(_gama)
    executor_vns_cnp.modify_gama(_gama)
    executor_vnd_cnp.modify_gama(_gama)
    executor_ns_cnp.modify_gama(_gama)


if __name__ != "__main__":
    pass
else:
    # set experiment parameter
    start, end, step = 0.01, 0.51, 0.01
    repeat_times = 20
    json_modifier = JsonModifier("../data/configure.json")

    executor_pre_tuned_alns_cnp = Executor(repeat_times, execute_pre_tuned_alns_cnp)
    executor_alns_cnp = Executor(repeat_times, execute_alns_cnp)
    executor_lns_cnp = Executor(repeat_times, execute_lns_cnp)
    executor_vns_cnp = Executor(repeat_times, execute_vns_cnp)
    executor_vnd_cnp = Executor(repeat_times, execute_vnd_cnp)
    executor_hpfs_cnp = Executor(repeat_times, execute_hpfs_cnp)
    executor_fcfs_cnp = Executor(repeat_times, execute_fcfs_cnp)
    executor_llf_cnp = Executor(repeat_times, execute_llf_cnp)
    executor_ns_cnp = Executor(repeat_times, execute_ns_cnp)

    json_modifier.modify_and_save({
        "Task_Number": 1500,
        "Satellite_Number": 5
    }
    )

    Result = {
        "pre_tuned_alns_cnp": [],
        "alns_cnp": [],
        "lns_cnp": [],
        "vns_cnp": [],
        "vnd_cnp": [],
        "ns_cnp": []
    }

    # evaluation system: benefits  runtimes total_benefits
    # create repeater and execute it for appointed times

    for gama in np.arange(start, end, step):

        modify_gama(gama)

        result_lns_cnp = run_and_analyse(executor_lns_cnp)
        Result["lns_cnp"].append(result_lns_cnp)

        result_alns_cnp = run_and_analyse(executor_alns_cnp)
        Result["alns_cnp"].append(result_alns_cnp)

        result_pre_tuned_alns_cnp = run_and_analyse(executor_pre_tuned_alns_cnp)
        Result["pre_tuned_alns_cnp"].append(result_pre_tuned_alns_cnp)

        result_vns_cnp = run_and_analyse(executor_vns_cnp)
        Result["vns_cnp"].append(result_vns_cnp)

        result_vnd_cnp = run_and_analyse(executor_vnd_cnp)
        Result["vnd_cnp"].append(result_vnd_cnp)

        result_ns_cnp = run_and_analyse(executor_ns_cnp)
        Result["ns_cnp"].append(result_ns_cnp)

    with open('../data/result_differ_gama.pkl', 'wb') as f:
        pickle.dump(Result, f)

