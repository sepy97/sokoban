from abc import ABC, abstractmethod

from sokoban.environment import SokobanState

def BaseHeuristic(ABC):
    @abstractmethod
    def __call__(self, state: SokobanState) -> float:
        pass