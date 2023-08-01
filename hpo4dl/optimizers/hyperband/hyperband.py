""" Implementation of : `Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization`  -
https://arxiv.org/abs/1603.06560
"""

from typing import List, Dict, Optional, Tuple
import math
import numpy as np
import pandas as pd

from hpo4dl.optimizers.abstract_optimizer import AbstractOptimizer
from hpo4dl.configuration_manager.abstract_configuration_manager import AbstractConfigurationManager


class HyperBand(AbstractOptimizer):
    """ Implementation of : `Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization`  -
          https://arxiv.org/abs/1603.06560
    """

    def __init__(
        self,
        max_budget: int,
        configuration_manager: AbstractConfigurationManager,
        eta: int = 3,
        seed: Optional[int] = None,
        minimize: bool = False,
        device: str = None,
    ):
        assert max_budget is not None, "Hyperband requires max_budget."
        self.max_budget: int = max_budget
        self.configuration_manager: AbstractConfigurationManager = configuration_manager
        self.eta: int = eta
        self.seed: int = seed
        self.minimize: bool = minimize
        self.device = device

        self.max_brackets = math.floor(math.log(self.max_budget) / math.log(self.eta))
        self.bracket_total_budget = (self.max_brackets + 1) * self.max_budget

        # Initialize bracket information
        self.bracket_max_rungs = np.arange(self.max_brackets + 1)[::-1]
        self.bracket_num_configs = \
            np.ceil(
                (self.bracket_total_budget * np.power(self.eta, self.bracket_max_rungs)) / (
                    self.max_budget * (self.bracket_max_rungs + 1))
            )
        self.bracket_fidelity = self.max_budget * np.power(float(self.eta), -1 * self.bracket_max_rungs)
        self.bracket_num_configs = self.bracket_num_configs.astype(int)
        self.bracket_fidelity = self.bracket_fidelity.astype(int)

        # Initialize successive halving information
        self.sh_num_config = []
        self.sh_fidelity = []
        self.sh_num_promotions = []

        for bracket_id, max_rungs in enumerate(self.bracket_max_rungs):
            num_configs = self.bracket_num_configs[bracket_id]
            fidelity = self.bracket_fidelity[bracket_id]
            self.sh_num_config.append([])
            self.sh_fidelity.append([])
            self.sh_num_promotions.append([])
            for i in range(max_rungs + 1):
                num_configs_i = math.floor(
                    num_configs * math.pow(self.eta, -1 * i)
                )
                num_configs_i = int(num_configs_i)
                fidelity_i = int(fidelity * math.pow(self.eta, i))
                num_promotions = int(math.floor(num_configs_i / self.eta))
                self.sh_num_config[bracket_id].append(num_configs_i)
                self.sh_fidelity[bracket_id].append(fidelity_i)
                self.sh_num_promotions[bracket_id].append(num_promotions)

        self.num_configurations: int = int(np.sum(self.bracket_num_configs))
        self.configuration_manager.add_configurations(num_configurations=self.num_configurations)

        self.history: pd.DataFrame = pd.DataFrame()
        self.bracket_history: pd.DataFrame = pd.DataFrame()

        self.current_bracket_level: int = 0
        self.current_rung_level: int = 0
        self.bracket_configuration_ids: List[int] = []

        end_id: int = self.bracket_num_configs[self.current_bracket_level]
        self.bracket_configuration_ids = list(range(end_id))

    def get_top_k_configuration_id(self, data: pd.DataFrame, k: int) -> List[int]:
        """ Returns the top 'k' configuration IDs based on the performance.

        Args:
            data (pd.DataFrame): A pandas DataFrame containing the performance data. The DataFrame
                should have 'configuration_id' and 'metric' columns.
            k (int): The number of best configuration IDs to return.

        Returns:
            List[int]: A list of best 'k' configuration IDs.
        """
        if self.minimize:
            grouped_data = data.groupby('configuration_id')['metric'].min().reset_index()
        else:
            grouped_data = data.groupby('configuration_id')['metric'].max().reset_index()
        sorted_bracket_history = grouped_data.sort_values('metric', ascending=self.minimize)
        top_k = sorted_bracket_history.head(k)['configuration_id']
        top_k = list(top_k)
        return top_k

    def suggest(self) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """ Suggests the next configuration indices and corresponding fidelities to evaluate.

        Returns:
            Tuple: A tuple where the first element is a list of configuration IDs and
                    the second element is a list of fidelities.
        """
        if self.current_bracket_level >= len(self.bracket_num_configs):
            return None, None

        current_r = self.sh_fidelity[self.current_bracket_level][self.current_rung_level]

        bracket_fidelities = [current_r] * len(self.bracket_configuration_ids)
        return self.bracket_configuration_ids, bracket_fidelities

    def observe(
        self,
        configuration_results: List[Dict]
    ) -> None:
        """ Observes the results of the configuration and fidelity evaluation.

        Args:
            configuration_results : List of results from configuration that were evaluated.
        """
        new_configuration_results = pd.DataFrame(configuration_results)
        self.bracket_history = pd.concat([self.bracket_history, new_configuration_results], axis=0, ignore_index=True)
        self.history = pd.concat([self.history, new_configuration_results], axis=0, ignore_index=True)

        current_num_promotions = self.sh_num_promotions[self.current_bracket_level][self.current_rung_level]
        self.bracket_configuration_ids = self.get_top_k_configuration_id(
            data=self.bracket_history,
            k=current_num_promotions
        )
        top_k_mask = self.bracket_history['configuration_id'].isin(self.bracket_configuration_ids)
        self.bracket_history = self.bracket_history[top_k_mask]

        self.set_next_iteration()

    def set_next_iteration(self) -> None:
        """ Advances to the next iteration.
        """
        self.current_rung_level += 1
        if self.current_rung_level > self.bracket_max_rungs[self.current_bracket_level]:
            self.current_bracket_level += 1
            self.current_rung_level = 0
            self.bracket_history = pd.DataFrame()
            if self.current_bracket_level < len(self.bracket_num_configs):
                start_id = np.sum(self.bracket_num_configs[:self.current_bracket_level])
                end_id = start_id + self.bracket_num_configs[self.current_bracket_level]
                self.bracket_configuration_ids = list(range(start_id, end_id))

    def get_best_configuration_id(self) -> int:
        """ Gets the index of the best configuration seen so far.

        Returns:
            int: ID of the best configuration.
        """
        best_configuration_id = self.get_top_k_configuration_id(data=self.history, k=1)
        return best_configuration_id[0]
