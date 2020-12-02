from .base import BaseHeuristic
from sokoban.environment import SokobanState

class GreedyHeuristic:
    """ A constant heuristic, turning A* into greedy best-first search. """
    def __call__(self, state: SokobanState) -> float:
        return 0