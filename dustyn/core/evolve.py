import abc

import numpy as np

from dustyn._typing import FloatLike


class EvolutionModel(abc.ABC):
    @abc.abstractmethod
    def evolve(self, t: float, R: np.ndarray, **kwargs) -> np.ndarray:
        """Compute the time evolution of a state vector R"""
        pass

    @abc.abstractmethod
    def get_max_timestep(
        self,
        R: np.ndarray,
    ) -> FloatLike:
        pass
