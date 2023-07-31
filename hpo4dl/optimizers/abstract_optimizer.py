""" Abstract base class for optimizers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Tuple


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
        configuration_id: List[int],
        fidelity: List[int],
        metric: List[Dict[str, Union[List, int, float, str]]]
    ) -> None:
        """ Observes the results of the configuration and fidelity evaluation.

        Args:
            configuration_id : List of configuration indices that were evaluated.
            fidelity : List of fidelities that were used.
            metric : Evaluation results for each configuration/fidelity pair.
        """

    @abstractmethod
    def get_best_configuration_id(self) -> int:
        """ Gets the index of the best configuration seen so far.

        Returns:
            int: ID of the best configuration.
        """

    @abstractmethod
    def close(self):
        """ Closes the optimizer.
        """
