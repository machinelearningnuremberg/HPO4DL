""" Tuner class which is responsible for hyperparameter tuning.
The Tuner uses various optimization strategies to find the best hyperparameters
for a given objective function.
"""

import warnings
import shutil
from pathlib import Path
from typing import Dict, List, Union, Callable
import ConfigSpace as CS

from hpo4dl.graybox_wrapper.abstract_graybox_wrapper import AbstractGrayBoxWrapper
from hpo4dl.configuration_manager.abstract_configuration_manager import AbstractConfigurationManager
from hpo4dl.graybox_wrapper.graybox_wrapper import GrayBoxWrapper
from hpo4dl.configuration_manager.configuration_manager import ConfigurationManager
from hpo4dl.optimizers.abstract_optimizer import AbstractOptimizer
from hpo4dl.optimizers.hyperband.hyperband import HyperBand


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
        max_total_budget: int,
        result_path: Union[str, Path],
        optimizer: Union[str, AbstractOptimizer] = 'hyperband',
        minimize: bool = False,
        seed: int = None,
        device: str = None,
    ):
        self.seed = seed
        self.device = device
        self.objective_function = objective_function
        self.configuration_space = configuration_space
        self.optimizer = optimizer
        self.is_minimize = minimize
        self.optimizer_budget = max_total_budget
        self.max_epochs = max_epochs
        self.current_optimizer_budget = 0
        self.num_configurations = 0
        self.best_configuration_info = {}
        self.result_path = Path(result_path)

        self.configuration_manager = ConfigurationManager(
            configuration_space=self.configuration_space,
            num_configurations=self.num_configurations,
            seed=self.seed
        )

        self.graybox_wrapper = GrayBoxWrapper(
            objective_function=self.objective_function,
            configuration_manager=self.configuration_manager,
        )

        if not isinstance(optimizer, str):
            self.surrogate = optimizer
        elif optimizer == 'hyperband':
            self.surrogate = HyperBand(
                max_budget=self.max_epochs,
                configuration_manager=self.configuration_manager,
                seed=self.seed,
                minimization=self.is_minimize,
                device=self.device,
            )
        else:
            raise ValueError(f"optimizer {optimizer} does not exist.")

    def run(self) -> Dict:
        """ Starts the optimization process.

        Returns:
            Dict: the best configuration found.

        """
        while self.optimizer_budget is None or self.current_optimizer_budget < self.optimizer_budget:
            configuration_indices, fidelities = self.surrogate.suggest()
            if configuration_indices is None:
                break

            metric = self.graybox_wrapper.start_trial(
                configuration_id=configuration_indices,
                epoch=fidelities
            )

            self.surrogate.observe(
                configuration_id=configuration_indices,
                fidelity=fidelities,
                metric=metric
            )

        best_configuration_id = self.surrogate.get_best_configuration_id()
        best_configuration = self.configuration_manager.get_configuration(configuration_id=best_configuration_id)
        self.set_best_model_checkpoint(configuration_id=best_configuration_id)

        self.surrogate.close()
        self.graybox_wrapper.close()

        return best_configuration

    def set_best_model_checkpoint(self, configuration_id: int) -> None:
        """ Sets the checkpoint of the best model.

        Args:
            configuration_id: The id of the configuration for the best model.

        """
        checkpoint_path = self.graybox_wrapper.get_checkpoint_path(configuration_id=configuration_id)
        if checkpoint_path.exists():
            shutil.copy2(checkpoint_path, self.result_path)
        else:
            warnings.warn("Best model checkpoint does not exist.", RuntimeWarning)
