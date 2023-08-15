""" Logger for saving configuration results.
"""

from pathlib import Path
import pandas as pd
from typing import List, Dict


class ResultLogger:
    """ Logger for saving configuration results.
    """

    def __init__(self, path: Path, minimize: bool = False):
        self.root_path = path
        self.minimize = minimize
        self.history_data: pd.DataFrame = pd.DataFrame()

    def add_configuration_results(self, configuration_results):
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
