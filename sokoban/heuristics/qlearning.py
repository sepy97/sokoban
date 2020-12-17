from .base import BaseHeuristic
from sokoban.environment import SokobanState, states_to_numpy

from typing import List
import numpy as np


def split_range(size: int):
        if size % 2 == 0:
            return size // 2, size // 2
        else:
            return size // 2, size // 2 + 1


class QLearningHeuristic(BaseHeuristic):
    def __init__(self, torch_file: str, max_size: int = 32, cuda: bool = False):
        super(QLearningHeuristic, self).__init__()

        import torch

        self.network = torch.jit.load(torch_file)
        if cuda:
            self.network = self.network.cuda()

        self.max_size = max_size
        self.torch = torch
        self.cuda = cuda

    def state_to_nnet_input(self, state):
        state = np.pad(state.map, list(map(split_range, self.max_size - np.array(state.map.shape))), constant_values=4)
        state = self.torch.from_numpy(state)
        return state.unsqueeze(0)

    def __call__(self, state: SokobanState) -> float:
        return self.network(self.state_to_nnet_input(state)).min().item()

    def batch_call(self, states: List[SokobanState]) -> List[float]:
        # states = self.torch.stack([self.state_to_nnet_input(state).squeeze() for state in states])
        states = states_to_numpy(states, self.max_size)
        states = self.torch.from_numpy(states)

        if self.cuda:
            states = states.cuda()

        heuristics =  self.network(states).min(1).values
        heuristics = heuristics.cpu().numpy()

        return heuristics
        