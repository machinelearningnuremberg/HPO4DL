import numpy as np
import pandas as pd
from pathlib import Path
import json


class MetricLogger:
    def __init__(self, configuration_manager, minimization=False):
        self.minimization = minimization
        self.configuration_manager = configuration_manager
        self.log_results_data = {
            "performance": [],
            "epoch": [],
            "current_best": [],
            "current_best_config_id": [],
            "current_best_config": [],
        }
        self.history: pd.DataFrame = pd.DataFrame()

    def add_observations(self, observations):
        best_performance = np.PINF if self.minimization else np.NINF
        best_config_id = -1
        for entry in observations:
            if self.minimization:
                if entry["performance"] < best_performance:
                    best_performance = entry["performance"]
                    best_config_id = entry["config_id"]
            else:
                if entry["performance"] > best_performance:
                    best_performance = entry["performance"]
                    best_config_id = entry["config_id"]

        new_history_entries = pd.DataFrame(observations)
        self.history = pd.concat([self.history, new_history_entries], axis=0)

        self.log_results_data["performance"].append(float(best_performance))
        best_epoch = np.max(new_history_entries["fidelity"])
        self.log_results_data["epoch"].append(int(best_epoch))

        if len(self.log_results_data["current_best"]) > 1:
            current_best = self.log_results_data["current_best"][-1]
            current_best_config_id = self.log_results_data["current_best_config_id"][-1]
        else:
            current_best = np.PINF if self.minimization else np.NINF
            current_best_config_id = -1

        if self.minimization:
            if best_performance < current_best:
                current_best = best_performance
                current_best_config_id = best_config_id
        else:
            if best_performance > current_best:
                current_best = best_performance
                current_best_config_id = best_config_id

        self.log_results_data["current_best"].append(float(current_best))
        self.log_results_data["current_best_config_id"].append(int(current_best_config_id))
        current_best_config = self.configuration_manager.get_configuration(configuration_id=current_best_config_id)
        self.log_results_data["current_best_config"].append(current_best_config)

    def log_results(self):
        log_root_path = Path("../hpo_results")
        log_root_path.mkdir(parents=True, exist_ok=True)
        save_path = log_root_path / "all_history.csv"
        self.history.to_csv(save_path)

        save_path = log_root_path / "curve.json"
        with open(save_path, 'w') as file:
            json.dump(self.log_results_data, file)
