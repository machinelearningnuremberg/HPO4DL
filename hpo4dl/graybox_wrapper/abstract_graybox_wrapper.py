""" Abstract base class for graybox wrapper.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Callable

from hpo4dl.utils.configuration_dataclasses import ConfigurationInfo
from hpo4dl.configuration_manager.abstract_configuration_manager import AbstractConfigurationManager


class AbstractGrayBoxWrapper(ABC):
    """ Abstract base class for graybox wrapper.
    """

    @abstractmethod
    def __init__(
        self,
        checkpoint_path: Path,
        objective_function: Callable,
        configuration_manager: AbstractConfigurationManager
    ):
        """ Initialize graybox wrapper.
z
        Args:
            checkpoint_path: path to save checkpoints.
            objective_function : A function that defines the optimization problem.
            configuration_manager : An instance of a manager that handles the various configurations
            associated with the optimization problem.
        """

    @abstractmethod
    def start_trial(
        self,
        configuration_info: List[ConfigurationInfo],
        epoch: List[int]
    ) -> List[Dict]:
        """ Evaluate a batch of configurations.

        Args:
            configuration_info: IDs of the configurations to be evaluated.
            epoch: The epochs to be evaluated.

        Returns:
            List[Dict]: Configuration results for all configuration/epoch pair.
        """

    @abstractmethod
    def get_checkpoint_path(self, configuration_id: int) -> Path:
        """ Gets the checkpoint path for the given configuration.

        Args:
            configuration_id: The ID of the configuration.

        Returns:
            Path: The checkpoint path.
        """

    @abstractmethod
    def close(self) -> None:
        """ Closes the wrapper and cleans up the checkpoint directory.
        """
