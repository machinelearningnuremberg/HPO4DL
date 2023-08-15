""" Abstract base class for optimizers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple


class AbstractOptimizer(ABC):
    """ Abstract base class for optimizers.
    """

    @abstractmethod
    def suggest(self) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """ Suggests the next configuration indices and corresponding fidelities to evaluate.

        Returns:
            Tuple: A tuple where the first element is a list of configuration IDs and
                    the second element is a list of fidelities.
        """

    @abstractmethod
    def observe(
        self,
        configuration_results: List[Dict]
    ) -> None:
        """ Observes the results of the configuration and fidelity evaluation.

        Args:
            configuration_results : List of results from configuration that were evaluated.
        """
