""" Tuner class which is responsible for hyperparameter tuning.
The Tuner uses various optimization strategies to find the best hyperparameters
for a given objective function.
"""
import time
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
from hpo4dl.utils.configuration_dataclasses import ConfigurationInfo


class Tuner:
    """ Manages optimization of Hyperparameters.
    """
    configuration_manager: AbstractConfigurationManager
    graybox_wrapper: AbstractGrayBoxWrapper
    surrogate: AbstractOptimizer

    def __init__(
        self,
        objective_function: Callable[[Dict, int, int, str], Union[List[Dict], Dict]],
        configuration_space: CS.ConfigurationSpace,
        max_epochs: int,
        max_budget: Optional[int],
        output_path: Union[str, Path],
        optimizer: Literal["hyperband", "dyhpo"] = 'dyhpo',
        minimize: bool = False,
        num_configurations: int = 1000,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        storage_path: Union[Path, str, None] = None,
    ):
        """ Initialize tuner.

        Args:
            objective_function: Function to evaluate a configuration's performance.
            configuration_space: Space of possible configurations for optimization.
            max_epochs: Maximum number of epochs for optimization of objective function.
            max_budget: Maximum budget for hyperparameter optimization surrogate in terms of epochs.
                        Not required when using hyperband.
            output_path: Path to store optimization results.
            optimizer: Optimization strategy to use. Choices: 'hyperband' or 'dyhpo'.
            minimize: If True, minimize the objective function, else maximize.
            num_configurations: Total number of configurations to sample.
            seed: Seed for reproducibility.
            device: Hardware device for execution. ('cuda', 'cpu').
            storage_path: Path to store temporary intermediate data.
        """
        assert optimizer in ["hyperband", "dyhpo"], "Only supported hyperband and dyhpo optimizers."
        assert optimizer == "hyperband" or \
               (optimizer != "hyperband" and max_budget is not None and max_budget > 0), \
            f"Max budget needs to be defined for {optimizer}."

        self.seed = seed
        self.device = device
        self.objective_function = objective_function
        self.configuration_space = configuration_space
        self.optimizer = optimizer
        self.minimize = minimize
        self.optimizer_max_budget = max_budget
        self.max_epochs = max_epochs
        self.current_optimizer_budget = 0
        self.num_configurations = num_configurations
        self.best_configuration_info: Dict = {}
        self.experiment_name = f'experiment_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        self.output_path = Path(output_path) / self.experiment_name

        self.configuration_manager = ConfigurationManager(
            configuration_space=self.configuration_space,
            num_configurations=self.num_configurations,
            seed=self.seed
        )

        if storage_path is None:
            storage_path = Path(os.path.expanduser('~/hpo4dl'))
        self.storage_path = Path(storage_path) / 'hpo4dl'
        checkpoint_path = self.storage_path / self.experiment_name

        self.graybox_wrapper = GrayBoxWrapper(
            checkpoint_path=checkpoint_path,
            objective_function=self.objective_function,
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
            self.surrogate = DyHPOOptimizer(
                max_epochs=self.max_epochs,
                total_budget=self.optimizer_max_budget,
                configuration_manager=self.configuration_manager,
                seed=self.seed,
                minimization=self.minimize,
                device=self.device,
                output_path=str(self.output_path),
                num_configurations=self.num_configurations,
                verbose=False,
            )
        else:
            raise ValueError(f"optimizer {optimizer} does not exist.")

        self.result_logger = ResultLogger(
            path=self.output_path,
            minimize=self.minimize,
            configuration_space=self.configuration_space,
        )

    def run(self) -> Dict:
        """ Starts the optimization process.

        Returns:
            Dict: the best configuration found.

        """
        while self.optimizer_max_budget is None or self.current_optimizer_budget < self.optimizer_max_budget:
            surrogate_overhead_time = 0.0

            surrogate_start_time = time.perf_counter()
            configuration_indices, fidelities = self.surrogate.suggest()
            surrogate_end_time = time.perf_counter()

            surrogate_overhead_time += (surrogate_end_time - surrogate_start_time)

            if configuration_indices is None or fidelities is None:
                break

            configuration_infos = []
            for configuration_id in configuration_indices:
                configuration = self.configuration_manager.get_configuration(configuration_id=configuration_id)
                configuration_info = ConfigurationInfo(
                    configuration=configuration,
                    configuration_id=configuration_id,
                )
                configuration_infos.append(configuration_info)

            configuration_results = self.graybox_wrapper.start_trial(
                configuration_info=configuration_infos,
                epoch=fidelities
            )

            surrogate_start_time = time.perf_counter()
            self.surrogate.observe(configuration_results=configuration_results)
            surrogate_end_time = time.perf_counter()

            surrogate_overhead_time += (surrogate_end_time - surrogate_start_time)

            single_entry_overhead_time = surrogate_overhead_time / len(configuration_results)

            # Add surrogate overhead time
            for entry in configuration_results:
                entry["overhead_time"] = single_entry_overhead_time
                entry["total_time"] = entry["time"] + single_entry_overhead_time

            self.result_logger.add_configuration_results(configuration_results=configuration_results)

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
            destination_file_path = self.output_path / 'checkpoints'
            shutil.copytree(checkpoint_path, destination_file_path)
        else:
            warnings.warn("Best model checkpoint does not exist.", RuntimeWarning)
