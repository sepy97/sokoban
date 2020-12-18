import ctypes
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch_spread import NetworkClient, mp_ctx as mp
from torch_spread.buffer import Buffer, raw_buffer_and_size, ShapeBufferType, DtypeBufferType
from torch_spread.buffer_queue import BufferRing

from qlearning.parameters import Parameters
from qlearning.sokoban_environment import SokobanEnvironment


class OrderedReplayBuffer(BufferRing):
    def __init__(self,
                 state_shapes: ShapeBufferType,
                 state_types: DtypeBufferType,
                 max_size: int):
        """ A ring buffer for storing a prioritized replay buffer. Used for Deep Q Learning.

        Parameters
        ----------
        state_shapes: Buffer Shape
            The shapes of a single state as a buffer
        state_types: Buffer Type
            The types of a single state as a buffer
        max_size: int
            Maximum number of unique samples to hold in this buffer.
        """
        buffer_shapes = {
            "states": state_shapes,
            "results": state_shapes,
            "distances": tuple(),
            "actions": tuple(),
            "terminals": tuple(),
            "priorities": tuple(),
            "discounts": tuple(),
            "discount_costs": tuple(),
        }

        buffer_types = {
            "states": state_types,
            "results": state_types,
            "distances": torch.float32,
            "actions": torch.int64,
            "terminals": torch.uint8,
            "priorities": torch.float32,
            "discounts": torch.float32,
            "discount_costs": torch.float32,
        }

        super(OrderedReplayBuffer, self).__init__(buffer_shapes, buffer_types, max_size)

        self.max_priority = mp.Value(ctypes.c_float, lock=False)
        self.max_priority.value = 1.0
        self.current_sample_index = 0

    @property
    def priorities(self) -> np.ndarray:
        current_size = self.size
        return self.buffer[:current_size]('priorities').numpy()

    def reset(self):
        super(OrderedReplayBuffer, self).reset()
        self.max_priority.value = 1.0
        self.current_sample_index = 0

    def put(self, buffer, size: int = None):
        buffer, size = raw_buffer_and_size(buffer, size)

        # Compute the priority for an incoming sample
        buffer['priorities'][:] = self.max_priority.value

        # Put it into the buffer
        super(OrderedReplayBuffer, self).put(buffer, size)

    def sample(self, num_samples: int = 32) -> Tuple[np.ndarray, Buffer, Tensor]:
        current_size = self.size
        assert num_samples <= current_size, f"Buffer is not large enough to provide {num_samples} samples"

        idx = np.arange(self.current_sample_index, self.current_sample_index + num_samples) % current_size
        self.current_sample_index = (self.current_sample_index + num_samples) % current_size

        batch = self.buffer[idx]
        weights = torch.ones(num_samples)

        return idx, batch, weights


class EpsilonGreedyClient(NetworkClient):
    def __init__(self, config: dict, batch_size: int, epsilon: float):
        """ An extension to the regular NetworkClient that provides Q-learning policy functions.
        Parameters
        ----------
        config: dict
            Client configuration from the network manager.
        batch_size: int
            Maximum number of states you're planning on predicting at once.
        epsilon: float
            Probability of performing a random action
        """
        super().__init__(config, batch_size)
        self.epsilon = epsilon

    def predict_with_batching(self, states) -> Tensor:
        state_size = states[0].shape[0]
        predictions = torch.empty(state_size, *self.output_buffer.shape[1:])
        for start_idx in range(0, state_size, self.batch_size):
            end_idx = min(start_idx + self.batch_size, state_size)
            predictions[start_idx:end_idx] = self.predict([state[start_idx:end_idx] for state in states])

        return predictions

    @staticmethod
    def _greedy_actions(q_values: Tensor) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        greedy_actions = q_values.min(dim=1).indices
        return greedy_actions.numpy()

    def q_values_to_action(self, q_values: Tensor) -> np.ndarray:
        num_states, num_actions = q_values.shape

        # Take the action with the minimum cost to go as greedy
        greedy_actions = self._greedy_actions(q_values)

        # Generate random actions
        random_actions = np.random.randint(low=0, high=num_actions, size=num_states, dtype=greedy_actions.dtype)

        # Select between greedy and random actions using epsilon-greedy policy
        epsilons = np.random.rand(num_states)
        epsilons = (epsilons < self.epsilon).astype(np.int64)
        return np.choose(epsilons, (greedy_actions, random_actions))

    def greedy_actions(self, states) -> np.ndarray:
        q_values = self.predict(states)
        return self._greedy_actions(q_values)

    def sample_actions(self, states) -> np.ndarray:
        """ Sample many actions at once (for vectorized environment). """
        q_values = self.predict(states)
        return self.q_values_to_action(q_values)

    def greedy_action_with_batching(self,
                                    states,
                                    return_values: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        q_values = self.predict_with_batching(states)
        greedy_actions = self._greedy_actions(q_values)
        if return_values:
            return greedy_actions, q_values.min(dim=1).values.numpy()
        else:
            return greedy_actions

    def sample_actions_with_batching(self, states) -> np.ndarray:
        """ Sample many actions at once, support for more states than batch size. """
        q_values = self.predict_with_batching(states)
        return self.q_values_to_action(q_values)


class DQNNetwork(nn.Module):
    def __init__(self, worker: bool):
        super(DQNNetwork, self).__init__()

        self.worker = worker
        self.env = SokobanEnvironment(48, "./walls", min_targets=1, max_targets=32)
        self.network = self.env.get_nnet_model()

    def forward(self, x: Buffer) -> Tensor:
        x, size = raw_buffer_and_size(x)
        return self.network(*x)

    def q_values(self, states: Buffer, actions: Tensor):
        return self.forward(states).gather(1, actions.unsqueeze(1)).squeeze()


def dqn_target(policy_network: DQNNetwork,
               target_network: DQNNetwork,
               sample: Buffer,
               hparams: Parameters,
               initial: bool) -> torch.Tensor:
    with torch.no_grad():
        # Gather the final state visited along a path
        policy_network.eval()
        final_states = sample("results", raw=True)

        # On the first iteration, we force the predicted q values to be the distance
        if initial:
            q_min = sample("distances") - hparams.n_step

        else:
            q_min = target_network(final_states)
            q_min = q_min.min(dim=1).values

        # Bellman loss target
        targets = q_min * sample("discounts") * (1 - sample("terminals"))
        targets = targets + sample("discount_costs")

        # Clamp targets to be from 0 to generation_distance + 1
        targets = torch.min(targets, sample("distances") + 1)
        targets = torch.clamp(targets, min=0)

    return targets
