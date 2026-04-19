from abc import ABC, abstractmethod
from typing import Any, Tuple, Union, Dict

class BaseAlgorithm(ABC):
    """
    Abstract base class for environment-agnostic JAX-native algorithms.
    Follows a Stable-Baselines 3 inspired interface.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def train(self, env: Any, total_steps: int):
        """
        Train the agent on the given environment for total_steps.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save the model parameters to the given path.
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Load the model parameters from the given path.
        """
        pass
