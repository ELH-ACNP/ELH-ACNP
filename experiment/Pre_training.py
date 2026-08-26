#!/usr/bin/env python3
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.ELHACNP_train import execute_pre_tuned_alns_cnp
import matplotlib.pyplot as plt
from scipy.spatial.distance import chebyshev
try:
    import scienceplots
except ImportError:
    scienceplots = None
import numpy as np
import json
import pickle
import random

if scienceplots is not None:
    plt.style.use(['science', 'ieee'])

class Executor:
    """
    Experiment Repeater
    """
    operator_weights: list

    def __init__(self, n, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.execute_times = n
        self.operator_weights = []

    def run_function_n_times(self):
        """
        run experiment repeat n times
        :return:
        """
        for _ in range(self.execute_times):
            operator_weight = self.func(*self.args, **self.kwargs)
            self.operator_weights.append(operator_weight)

    def analyse_result(self):
        """
        calculate the mean of operator weights
        :return:
        """
        combined_array = np.stack(self.operator_weights)
        mean_operator_weights = np.mean(combined_array, axis=0)
        return mean_operator_weights

    def reset_container(self):
        """
        reset result container
        :return:
        """
        self.operator_weights = []

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

class BreakLoop(Exception):pass

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


def main():
    project_root = Path(__file__).resolve().parents[1]
    experiment_dir = Path(__file__).resolve().parent
    os.chdir(experiment_dir)

    config_path = project_root / "data" / "configure.json"
    weight_path = project_root / "data" / "weight_trained.pkl"
    curve_path = project_root / "data" / "weight_trained_curve.pkl"
    json_modifier = JsonModifier(str(config_path))
    original_config = json_modifier.load_data()

    satellite_scales = [1, 2, 3, 4, 5, 6]
    task_scales = [200, 400, 500, 600, 800, 1000, 1200, 1500]
    scenarios = [
        (satellite_number, task_number, 10_000 + index)
        for index, (satellite_number, task_number) in enumerate(
            (s, t) for s in satellite_scales for t in task_scales
        )
    ]

    alpha = 0.1
    minimum_update_error = 0.001
    maximum_updates = 5000  # Safety guard; convergence is controlled by the error threshold.
    scenario_rng = np.random.default_rng(3845)
    executor = Executor(1, execute_pre_tuned_alns_cnp, 0.24, 5, 1)
    error_curve = []

    # J4 denotes equal probabilities for the 16 strategy combinations.
    initial_weight = np.full((4, 4), 1 / 16, dtype=float)
    with weight_path.open("wb") as stream:
        pickle.dump(initial_weight, stream)

    previous_task_seed = os.environ.get("SATELLITE_TASK_SEED")
    previous_algorithm_seed = os.environ.get("SATELLITE_ALGORITHM_SEED")
    try:
        update_index = 0
        converged = False
        while update_index < maximum_updates and not converged:
            scenario_rng.shuffle(scenarios)
            for satellite_number, task_number, task_seed in scenarios:
                update_index += 1
                print(f"Training update {update_index}")

                with weight_path.open("rb") as stream:
                    weight_before = pickle.load(stream)

                json_modifier.modify_and_save({
                    "Task_Number": task_number,
                    "Satellite_Number": satellite_number,
                })
                algorithm_seed = 50_000 + update_index
                os.environ["SATELLITE_TASK_SEED"] = str(task_seed)
                os.environ["SATELLITE_ALGORITHM_SEED"] = str(algorithm_seed)
                random.seed(algorithm_seed)
                np.random.seed(algorithm_seed)

                learned_weight = run_and_analyse(executor)
                weight_after = weight_before + alpha * (learned_weight - weight_before)
                update_error = float(chebyshev(weight_after.ravel(), weight_before.ravel()))
                error_curve.append(update_error)
                print(f"update error = {update_error:.8f}")

                with weight_path.open("wb") as stream:
                    pickle.dump(weight_after, stream)

                if update_error <= minimum_update_error:
                    converged = True
                    break
                if update_index >= maximum_updates:
                    break

        if not converged:
            raise RuntimeError(
                f"Pre-training did not reach {minimum_update_error} within {maximum_updates} updates"
            )
    finally:
        json_modifier.write_data(original_config)
        if previous_task_seed is None:
            os.environ.pop("SATELLITE_TASK_SEED", None)
        else:
            os.environ["SATELLITE_TASK_SEED"] = previous_task_seed
        if previous_algorithm_seed is None:
            os.environ.pop("SATELLITE_ALGORITHM_SEED", None)
        else:
            os.environ["SATELLITE_ALGORITHM_SEED"] = previous_algorithm_seed

    with curve_path.open("wb") as stream:
        pickle.dump(error_curve, stream)


if __name__ == "__main__":
    main()

