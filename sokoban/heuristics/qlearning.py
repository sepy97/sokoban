from .base import BaseHeuristic
from sokoban.environment import SokobanState

import numpy as np


def split_range(size: int):
        if size % 2 == 0:
            return size // 2, size // 2
        else:
            return size // 2, size // 2 + 1


class QLearningHeuristic:
    def __init__(self, torch_file: str, max_size: int = 32):
        import torch

        self.network = torch.jit.load(torch_file)
        self.max_size = max_size
        self.torch = torch

    def state_to_nnet_input(self, state):
        state = np.pad(state.map, list(map(split_range, self.max_size - np.array(state.map.shape))), constant_values=4)
        state = self.torch.from_numpy(state)
        return state.unsqueeze(0)

    def __call__(self, state: SokobanState) -> float:
        return self.network(self.state_to_nnet_input(state)).min().item()