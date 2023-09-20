""" Abstract base class for configuration manager.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List
import ConfigSpace as CS
import pandas as pd


class AbstractConfigurationManager(ABC):
    """ Abstract base class for configuration manager.
    """

    @abstractmethod
    def __init__(
        self,
        configuration_space: CS.ConfigurationSpace,
        num_configurations: int = 0,
        seed: Optional[int] = None
    ):
        pass

    @abstractmethod
    def get_configurations(self) -> pd.DataFrame:
        """ Returns all generated configurations as a DataFrame.
        """

    @abstractmethod
    def get_configuration(self, configuration_id: int) -> Dict:
        """ Returns the configuration corresponding to the given ID.
        """

    @abstractmethod
    def add_configurations(self, num_configurations: int, max_try_limit: int = 100) -> int:
        """ Adds a specified number of unique configurations to the list.

        Args:
            num_configurations: number of configurations to add.
            max_try_limit: maximum number of tries to get non-duplicate configurations.

        Returns:
            int: Number of configurations added.
        """

    def set_configuration(self, configuration):
        """ Add the configuration to the list.

        Args:
            configuration: configuration to add.
        """

    @abstractmethod
    def get_log_indicator(self):
        """ Get log indicator for configuration space.
        """

    @abstractmethod
    def get_categorical_indicator(self) -> Tuple[List, Dict]:
        """ Get categorical indicator for configuration space.
        """
