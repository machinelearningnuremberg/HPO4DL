""" Wraps and evaluate the objective function and manages checkpoints for each trial.
"""

from pathlib import Path
from typing import List, Dict, Union, Callable
from datetime import datetime
import time
import shutil

from hpo4dl.graybox_wrapper.abstract_graybox_wrapper import AbstractGrayBoxWrapper
from hpo4dl.configuration_manager.abstract_configuration_manager import AbstractConfigurationManager


class GrayBoxWrapper(AbstractGrayBoxWrapper):
    """ Wraps and evaluate the objective function and manages checkpoints for each trial.
    """

    def __init__(self, objective_function: Callable, configuration_manager: AbstractConfigurationManager):
        self.objective_function = objective_function
        self.configuration_manager = configuration_manager
        self.previous_fidelities = {}
        self.checkpoint_paths = {}
        self.root_path = Path('../checkpoints') / f'experiment_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        self.trial_results = {}

    def start_trial(
        self, configuration_id: List[int],
        epoch: List[int]
    ) -> List[Dict[str, Union[List, int, float, str]]]:
        """ Evaluate a batch of configurations.

        Args:
            configuration_id: IDs of the configurations to be evaluated.
            epoch: The epochs to be evaluated.

        Returns:
            List[Dict[str, Union[List, int, float, str]]]: Metrics for each configuration/epoch pair.
        """
        metrics = []
        for trial_config_id, trial_epoch in zip(configuration_id, epoch):
            metric = self._run(configuration_id=trial_config_id, epoch=trial_epoch)
            metrics.append(metric)

        return metrics

    def _run(self, configuration_id: int, epoch: int) -> Dict:
        """ Evaluate a configuration.

        Args:
            configuration_id: ID of the configuration to be evaluated.
            epoch: The epoch to be evaluated.

        Returns:
            Dict: Metrics for the given configuration/epoch pair.
        """
        if (configuration_id, epoch) in self.trial_results:
            return self.trial_results[(configuration_id, epoch)]

        configuration = self.configuration_manager.get_configuration(configuration_id=configuration_id)

        if configuration_id in self.checkpoint_paths:
            checkpoint_path = self.checkpoint_paths[configuration_id]
        else:
            checkpoint_path = self.get_checkpoint_path(configuration_id)

        if configuration_id in self.previous_fidelities:
            previous_epoch = self.previous_fidelities[configuration_id]
        else:
            previous_epoch = 0

        start_time = time.perf_counter()
        metric = self.objective_function(
            configuration=configuration, epoch=epoch, previous_epoch=previous_epoch, checkpoint_path=checkpoint_path
        )
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        if not isinstance(metric, List):
            metric = [metric]

        metrics = {
            'metric': metric,
            'time': execution_time,
        }
        self.previous_fidelities[configuration_id] = epoch
        self.trial_results[(configuration_id, epoch)] = metrics

        return metrics

    def get_checkpoint_path(self, configuration_id: int) -> Path:
        """ Gets the checkpoint path for the given configuration.

        Args:
            configuration_id: The ID of the configuration.

        Returns:
            Path: The checkpoint path.
        """
        return self.root_path / f'trial_{configuration_id}' / 'last.pth.tar'

    def close(self) -> None:
        """ Closes the wrapper and cleans up the checkpoint directory.
        """
        if self.root_path.exists():
            shutil.rmtree(self.root_path)
