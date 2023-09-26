"""
Data classes for representing configuration details and results.
"""

from dataclasses import dataclass
from typing import Dict, Union


@dataclass
class ConfigurationResult:
    """Holds the results of a specific configuration evaluation."""
    configuration: Dict
    configuration_id: int
    epoch: int
    metric: Union[int, float]
    time: float


@dataclass
class ConfigurationInfo:
    """Basic details about a specific configuration."""
    configuration: Dict
    configuration_id: int
