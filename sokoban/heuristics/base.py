from abc import ABC, abstractmethod

from typing import List
from sokoban.environment import SokobanState


class BaseHeuristic(ABC):
    @abstractmethod
    def __call__(self, state: SokobanState) -> float:
        pass

    def batch_call(self, states: List[SokobanState]) -> List[float]:
        return [self(state) for state in states]
