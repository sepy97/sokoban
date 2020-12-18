from .base import BaseHeuristic
from sokoban.environment import SokobanState, states_to_numpy, states_to_numpy_partial

from typing import List
import numpy as np


class QLearningHeuristic(BaseHeuristic):
    def __init__(self, torch_file: str, max_size: int = 32, full_input: bool = True, cuda: bool = False):
        super(QLearningHeuristic, self).__init__()

        import torch

        self.network = torch.jit.load(torch_file)
        if cuda:
            self.network = self.network.cuda()

        self.full_input = full_input
        self.max_size = max_size
        self.torch = torch
        self.cuda = cuda

    def __call__(self, state: SokobanState) -> float:
        if self.full_input:
            states = states_to_numpy(np.array([state]), self.max_size, 0)
        else:
            states = [states_to_numpy_partial(np.array([state]), self.max_size)]

        states = [self.torch.from_numpy(x) for x in states]

        if self.cuda:
            states = [x.cuda() for x in states]

        return self.network(*states).min().item()

    def batch_call(self, states: List[SokobanState]) -> List[float]:
        if self.full_input:
            states = states_to_numpy(states, self.max_size, 0)
        else:
            states = states_to_numpy_partial(states, self.max_size)

        states = [self.torch.from_numpy(x) for x in states]

        if self.cuda:
            states = [x.cuda() for x in states]

        heuristics = self.network(*states).min(1).values
        heuristics = heuristics.cpu().numpy()

        return heuristics
