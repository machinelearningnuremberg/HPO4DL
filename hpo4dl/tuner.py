""" Tuner class which is responsible for hyperparameter tuning.
The Tuner uses various optimization strategies to find the best hyperparameters
for a given objective function.
"""

import warnings
import shutil
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Callable, Literal, Optional
import ConfigSpace as CS

from hpo4dl.graybox_wrapper.abstract_graybox_wrapper import AbstractGrayBoxWrapper
from hpo4dl.configuration_manager.abstract_configuration_manager import AbstractConfigurationManager
from hpo4dl.graybox_wrapper.graybox_wrapper import GrayBoxWrapper
from hpo4dl.configuration_manager.configuration_manager import ConfigurationManager
from hpo4dl.optimizers.abstract_optimizer import AbstractOptimizer
from hpo4dl.optimizers.hyperband.hyperband import HyperBand
from hpo4dl.optimizers.dyhpo.dyhpo_optimizer import DyHPOOptimizer
from hpo4dl.utils.result_logger import ResultLogger


class Tuner:
    """ Manages optimization of Hyperparameters.
    """
    configuration_manager: AbstractConfigurationManager
    graybox_wrapper: AbstractGrayBoxWrapper
    surrogate: AbstractOptimizer

    def __init__(
        self,
        objective_function: Callable[[Dict, int, int, Path], Union[List, int, float]],
        configuration_space: CS.ConfigurationSpace,
        max_epochs: int,
        max_total_budget: Optional[int],
        result_path: Union[str, Path],
        optimizer: Literal["hyperband", "dyhpo"] = 'hyperband',
        minimize: bool = False,
        seed: int = None,
        device: str = None,
    ):
        self.seed = seed
        self.device = device
        self.objective_function = objective_function
        self.configuration_space = configuration_space
        self.optimizer = optimizer
        self.minimize = minimize
        self.optimizer_budget = max_total_budget
        self.max_epochs = max_epochs
        self.current_optimizer_budget = 0
        self.num_configurations = 0
        self.best_configuration_info = {}
        self.experiment_name = f'experiment_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        self.result_path = Path(result_path) / self.experiment_name

        self.configuration_manager = ConfigurationManager(
            configuration_space=self.configuration_space,
            num_configurations=self.num_configurations,
            seed=self.seed
        )

        checkpoint_path = Path(os.path.expanduser('~/hpo4dl')) / self.experiment_name
        self.graybox_wrapper = GrayBoxWrapper(
            checkpoint_path=checkpoint_path,
            objective_function=self.objective_function,
            configuration_manager=self.configuration_manager,
        )

        if optimizer == 'hyperband':
            self.surrogate = HyperBand(
                max_budget=self.max_epochs,
                configuration_manager=self.configuration_manager,
                seed=self.seed,
                minimize=self.minimize,
                device=self.device,
            )
        elif optimizer == 'dyhpo':
            num_configurations = 1000
            self.surrogate = DyHPOOptimizer(
                max_epochs=self.max_epochs,
                total_budget=self.optimizer_budget,
                configuration_manager=self.configuration_manager,
                seed=self.seed,
                minimization=self.minimize,
                device=self.device,
                output_path=str(self.result_path),
                num_configurations=num_configurations,
            )
        else:
            raise ValueError(f"optimizer {optimizer} does not exist.")

        self.result_logger = ResultLogger(
            path=self.result_path,
            minimize=self.minimize
        )

    def run(self) -> Dict:
        """ Starts the optimization process.

        Returns:
            Dict: the best configuration found.

        """
        while self.optimizer_budget is None or self.current_optimizer_budget < self.optimizer_budget:
            configuration_indices, fidelities = self.surrogate.suggest()
            if configuration_indices is None:
                break

            configuration_results = self.graybox_wrapper.start_trial(
                configuration_id=configuration_indices,
                epoch=fidelities
            )
            self.result_logger.add_configuration_results(configuration_results=configuration_results)

            self.surrogate.observe(configuration_results=configuration_results)

        best_configuration_info = self.result_logger.get_best_configuration()

        # move best model checkpoint to result path
        self.set_best_model_checkpoint(configuration_id=best_configuration_info["configuration_id"])

        self.close()

        return best_configuration_info

    def close(self):
        """ Close tuner.
        """
        self.result_logger.save_results()
        self.graybox_wrapper.close()

    def set_best_model_checkpoint(self, configuration_id: int) -> None:
        """ Sets the checkpoint of the best model.

        Args:
            configuration_id: The id of the configuration for the best model.

        """
        checkpoint_path = self.graybox_wrapper.get_checkpoint_path(configuration_id=configuration_id)
        checkpoint_path = checkpoint_path.parent
        if checkpoint_path.exists():
            destination_file_path = self.result_path / 'checkpoints'
            shutil.copytree(checkpoint_path, destination_file_path)
        else:
            warnings.warn("Best model checkpoint does not exist.", RuntimeWarning)
