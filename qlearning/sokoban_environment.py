from glob import glob
from typing import List, Union, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sokoban.environment import SokobanState, states_to_numpy


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, size: int, stride: int):
        super().__init__()

        self.stride = stride

        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, size)
        self.activation = nn.PReLU(out_channels)

        self.pool = nn.Identity()
        if stride > 0:
            self.pool = nn.MaxPool2d(size, stride)

    def forward(self, x):
        return self.pool(self.activation(self.conv(self.norm(x))))


class LinearBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.norm = nn.BatchNorm1d(in_channels)
        self.conv = nn.Linear(in_channels, out_channels)
        self.activation = nn.PReLU(out_channels)

    def forward(self, x):
        return self.activation(self.conv(self.norm(x)))


class SokobanNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            ConvolutionBlock(5, 128, 5, 3),
            ConvolutionBlock(128, 128, 3, 3),
            ConvolutionBlock(128, 128, 4, 0),
        )

        self.block_embedding = nn.Sequential(
            LinearBlock(2, 8),
            LinearBlock(8, 16),
            LinearBlock(16, 32),
            LinearBlock(32, 64),
            LinearBlock(64, 128),
        )

        self.target_embedding = nn.Sequential(
            LinearBlock(2, 8),
            LinearBlock(8, 16),
            LinearBlock(16, 32),
            LinearBlock(32, 64),
            LinearBlock(64, 128),
        )

        self.encoder = nn.Transformer(128, 4, 6, 6, 128, 0.0)

        self.linear = nn.Sequential(
            LinearBlock(256, 128),
            LinearBlock(128, 64),
            LinearBlock(64, 32),
            LinearBlock(32, 16)
        )

        self.output = nn.Linear(16, 4)

    def forward(self, map, targets, blocks, mask):
        batch_size, num_targets, _ = targets.shape

        map = F.one_hot(map.long()).float().transpose(1, 3)
        map = self.conv(map)
        map = map.view(batch_size, -1)

        targets = self.target_embedding(targets.view(batch_size * num_targets, 2).float())
        blocks = self.block_embedding(blocks.view(batch_size * num_targets, 2).float())

        targets = targets.view(batch_size, num_targets, -1)
        blocks = blocks.view(batch_size, num_targets, -1)

        targets = targets * ~mask.unsqueeze(-1)
        blocks = blocks * ~mask.unsqueeze(-1)

        targets = targets.permute(1, 0, 2)
        blocks = blocks.permute(1, 0, 2)

        targets = self.encoder(targets, blocks, src_key_padding_mask=mask, tgt_key_padding_mask=mask)
        targets = targets.mean(0)

        x = torch.cat([map, targets], -1)

        return self.output(self.linear(x))


def split_range(size: int) -> Tuple[int, int]:
    if size % 2 == 0:
        return size // 2, size // 2
    else:
        return size // 2, size // 2 + 1


class SokobanEnvironment:
    def __init__(self, max_size: int, wall_directory: str, min_targets: int = 2, max_targets: int = 5):
        super(SokobanEnvironment, self).__init__()

        self.max_size = max_size
        self.wall_files = np.array(glob(f"{wall_directory}/*.txt"))
        self.max_targets = max_targets
        self.min_targets = min_targets

    @property
    def num_actions_max(self):
        return 4

    def next_state(self, states: List[SokobanState], actions: Union[List[int], np.ndarray]) -> Tuple[
        List[SokobanState], List[float]]:
        new_states = [state.next_state(action) for state, action in zip(states, actions)]
        transition_costs = np.ones(len(states), np.float32)

        return new_states, transition_costs

    def rand_action(self, states: List[SokobanState]) -> Union[List[int], np.ndarray]:
        actions = np.random.randint(0, self.num_actions_max, size=len(states))
        return actions

    def generate_goal_states(self, num_states: int) -> List[SokobanState]:
        wall_files = np.random.choice(self.wall_files, size=num_states)
        num_targets = np.random.randint(self.min_targets, self.max_targets + 1, size=num_states)

        return [SokobanState.generate(str(file), num, 0)
                for file, num in zip(wall_files, num_targets)]

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[
        List[SokobanState], List[int]]:
        wall_files = np.random.choice(self.wall_files, size=num_states)
        num_targets = np.random.randint(self.min_targets, self.max_targets + 1, size=num_states)
        steps = np.random.randint(backwards_range[0], backwards_range[1] + 1, size=num_states)

        states = [SokobanState.generate(str(file), num, 2 * step)
                  for file, num, step in zip(wall_files, num_targets, steps)]

        return states, steps

    def is_solved(self, states: List[SokobanState]) -> np.array:
        return np.asarray([state.solved for state in states])

    def state_to_nnet_input(self, states: List[SokobanState]) -> List[np.ndarray]:
        return states_to_numpy(np.asarray(states), self.max_size, self.max_targets)

    def get_num_moves(self) -> int:
        return 4

    def get_nnet_model(self) -> nn.Module:
        return SokobanNetwork()

    def state_information(self):
        """ Get the shape shape information for this environment.
            Feel free to overwrite with a more efficient option.
        """
        example_states = [v for v in self.state_to_nnet_input(self.generate_goal_states(1))]
        state_shapes = [state.shape[1:] for state in example_states]
        state_types = [state.dtype for state in example_states]

        return state_shapes, state_types

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
