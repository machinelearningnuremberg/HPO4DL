""" Logger for saving configuration results.
"""

from pathlib import Path
from typing import List, Dict
import json
import pandas as pd
import ConfigSpace as CS


class ResultLogger:
    """ Logger for saving configuration results.
    """

    def __init__(self, path: Path, minimize: bool = False, configuration_space=None):
        self.root_path = path
        self.minimize = minimize
        self.history_data: pd.DataFrame = pd.DataFrame()
        self.root_path.mkdir(parents=True, exist_ok=True)
        if configuration_space is not None:
            self.save_configuration_space(configuration_space=configuration_space)

    def save_configuration_space(self, configuration_space: CS.ConfigurationSpace):
        """ Save configuration space details as a json.

        Args:
            configuration_space: Hyperparameter space.

        """
        hyperparameters = []
        for hparam in list(configuration_space.values()):
            hparam_dict = {
                "name": hparam.name,
                "type": type(hparam).__name__,
                "default": hparam.default_value,
                "values": None if not hasattr(hparam, "choices") else list(hparam.choices),
                "bounds": None if not hasattr(hparam, "lower") else [hparam.lower, hparam.upper],
                "log": None if not hasattr(hparam, "log") else hparam.log
            }
            hyperparameters.append(hparam_dict)

        with open(self.root_path / 'configuration_space.json', 'w') as file_handle:
            json.dump(hyperparameters, file_handle, indent=4)

    def add_configuration_results(self, configuration_results: List[Dict]):
        """ Add configuration results to result log.

        Args:
            configuration_results: List of results from configuration that were evaluated.
        """
        new_configuration_results = pd.DataFrame(configuration_results)
        self.history_data = pd.concat([self.history_data, new_configuration_results], axis=0, ignore_index=True)

        self.save_results()

    def process_results(self):
        """ Calculate best performance.
        """
        if self.minimize:
            self.history_data['best_metric'] = self.history_data['metric'].cummin()
        else:
            self.history_data['best_metric'] = self.history_data['metric'].cummax()

    def save_results(self):
        """ Save hyperparameter optimization results to disk.
        """
        self.process_results()
        self.root_path.mkdir(parents=True, exist_ok=True)
        save_path = self.root_path / "hpo4dl_results.csv"
        self.history_data.to_csv(save_path)

    def get_best_configuration(self) -> Dict:
        """ Gets the index of the best configuration seen so far.

        Returns:
            Dict: Information of the best configuration.
        """
        if self.minimize:
            best_configuration = self.history_data.loc[self.history_data['metric'].idxmin()]
        else:
            best_configuration = self.history_data.loc[self.history_data['metric'].idxmax()]
        return best_configuration.to_dict()
