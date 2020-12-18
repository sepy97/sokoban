import ctypes
from typing import Tuple, List, Union, Optional

import numpy as np
import torch
import os
from torch_spread import mp_ctx

from qlearning.sokoban_environment import SokobanEnvironment
from qlearning.parameters import Parameters
from qlearning.qlearning import EpsilonGreedyClient
from qlearning.output_manager import StatusBarUpdater, AsyncStatusBar


class GBFSWorker(mp_ctx.Process):
    KILL = b'KILL'
    EVALUATE = b'EVALUATE'

    def __init__(self,
                 hparams: Parameters,
                 client_config: dict,
                 progress_bar: StatusBarUpdater,
                 request_queue: mp_ctx.JoinableQueue,
                 buffer_lock: mp_ctx.Lock,
                 quantity_buffer: mp_ctx.Array,
                 solved_buffer: mp_ctx.Array,
                 step_buffer: mp_ctx.Array,
                 cost_buffer: mp_ctx.Array):
        super(GBFSWorker, self).__init__()

        self.hparams = hparams
        self.client_config = client_config
        self.progress_bar = progress_bar
        self.request_queue = request_queue
        self.buffer_lock = buffer_lock

        self.quantity_buffer = quantity_buffer
        self.solved_buffer = solved_buffer
        self.step_buffer = step_buffer
        self.cost_buffer = cost_buffer

        self.ready_event = mp_ctx.Event()

    @staticmethod
    def state_vectors(env: SokobanEnvironment, states) -> List[torch.Tensor]:
        """ Helper function to convert state objects into PyTorch tensors. """
        return [torch.from_numpy(v) for v in env.state_to_nnet_input(states)]

    @staticmethod
    def queue(request_queue, num_states: int, backwards_range: Tuple[int, int]):
        """ Queue up an evaluation run on dynamically generated starting states. """
        request_queue.put((GBFSWorker.EVALUATE, num_states, backwards_range))

    def shutdown(self):
        self.request_queue.put((self.KILL,))

    def run(self):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

        if hasattr(self, "pid"):
            torch.manual_seed(self.pid)
            np.random.seed(self.pid)

        env = SokobanEnvironment(48, "./walls", min_targets=1, max_targets=32)
        max_steps = len(self.quantity_buffer)

        with EpsilonGreedyClient(self.client_config, self.hparams.simulation_network_batch_size, 0.0) as client:
            self.ready_event.set()

            while True:
                command, *data = self.request_queue.get()

                # Option 1: Kill switch
                if command == self.KILL:
                    self.request_queue.task_done()
                    return

                # Option 2: Perform an evaluation
                with env:
                    num_states, backwards_range = data

                    # Setup the initial states to begin testing from
                    states, distances = env.generate_states(num_states, backwards_range)
                    total_steps = np.zeros(num_states, dtype=np.int64)
                    terminals = env.is_solved(states)
                    initial_costs = None

                    # Perform GBFS Rollout
                    for step in range(self.hparams.testing_max_actions):
                        # Sample the neural network for an action
                        state_vectors = self.state_vectors(env, states)
                        if initial_costs is None:
                            actions, initial_costs = client.greedy_action_with_batching(state_vectors, True)
                        else:
                            actions = client.greedy_action_with_batching(state_vectors)

                        # Perform a simulation step
                        states, _ = env.next_state(states, actions)

                        # Add a new step to any non-terminal transitions
                        total_steps += ~terminals

                        # Update terminals to keep track of any states that have been solved
                        terminals |= env.is_solved(states)

                        # If every environment has finished, then break out early
                        if np.all(terminals):
                            break

                    # Perform result aggregation locally to avoid using the locks too much
                    local_quantity_buffer = np.zeros(max_steps, dtype=np.int64)
                    local_solved_buffer = np.zeros(max_steps, dtype=np.int64)
                    local_step_buffer = np.zeros(max_steps, dtype=np.int64)
                    local_cost_buffer = np.zeros(max_steps, dtype=np.float32)
                    for distance, solved, steps, costs in zip(distances, terminals, total_steps, initial_costs):
                        local_quantity_buffer[distance] += 1
                        local_cost_buffer[distance] += costs

                        if solved:
                            local_solved_buffer[distance] += 1
                            local_step_buffer[distance] += steps

                    # Update the results buffer with our simulation
                    with self.buffer_lock:
                        for distance in range(max_steps):
                            self.quantity_buffer[distance] += local_quantity_buffer[distance]
                            self.solved_buffer[distance] += local_solved_buffer[distance]
                            self.step_buffer[distance] += local_step_buffer[distance]
                            self.cost_buffer[distance] += local_cost_buffer[distance]

                    self.progress_bar.update(num_states)
                    self.request_queue.task_done()


class GBFSEvaluator:
    def __init__(self, hparams: Parameters, client_config: dict):
        self.hparams = hparams
        self.client_config = client_config

        self.request_queue = mp_ctx.JoinableQueue()
        self.workers: Optional[List[GBFSWorker]] = None
        self.progress_bar = AsyncStatusBar(enable=self.hparams.progress_bar)

        self._quantity_buffer = mp_ctx.Array(ctypes.c_int64, hparams.testing_max_steps + 1, lock=False)
        self.quantity_buffer = np.frombuffer(self._quantity_buffer, dtype=np.int64)

        self._solved_buffer = mp_ctx.Array(ctypes.c_int64, hparams.testing_max_steps + 1, lock=False)
        self.solved_buffer = np.frombuffer(self._solved_buffer, dtype=np.int64)

        self._step_buffer = mp_ctx.Array(ctypes.c_int64, hparams.testing_max_steps + 1, lock=False)
        self.step_buffer = np.frombuffer(self._step_buffer, dtype=np.int64)

        self._cost_buffer = mp_ctx.Array(ctypes.c_float, hparams.testing_max_steps + 1, lock=False)
        self.cost_buffer = np.frombuffer(self._cost_buffer, dtype=np.float32)

        self._buffers = (self._quantity_buffer, self._solved_buffer, self._step_buffer, self._cost_buffer)
        self.buffers = (self.quantity_buffer, self.solved_buffer, self.step_buffer, self.cost_buffer)

        self.buffer_lock = mp_ctx.Lock()

    def clear_buffers(self):
        for buffer in self.buffers:
            buffer[:] = 0

    def start(self, wait_for_start: bool = True):
        if self.workers is not None:
            raise AssertionError("Cannot start the evaluator twice.")

        print("Starting Evaluator")

        self.workers = []
        for _ in range(self.hparams.testing_workers):
            worker = GBFSWorker(self.hparams, self.client_config, self.progress_bar.updater, self.request_queue,
                                self.buffer_lock, *self._buffers)
            worker.start()
            self.workers.append(worker)

        if wait_for_start:
            for worker in self.workers:
                worker.ready_event.wait()

        print("Evaluator Ready")

    def stop(self):
        if self.workers is None:
            raise AssertionError("Cannot stop a evaluator that isn't started")

        for worker in self.workers:
            worker.shutdown()

        for worker in self.workers:
            worker.join(timeout=1)

        self.request_queue.join()
        self.workers = None

    def evaluate(self, num_states: int, backwards_range: Union[Tuple[int, int], int]):
        if self.workers is None:
            raise AssertionError("Cannot collect until collector is started.")

        if not isinstance(backwards_range, (list, tuple)):
            backwards_range = (0, backwards_range)

        # Clear the buffers from last evaluations
        self.clear_buffers()

        with self.progress_bar("Evaluating", num_states):
            # Queue up all of the batches
            worker_batch_size = num_states // len(self.workers)
            for start_state in range(0, num_states, worker_batch_size):
                end_state = min(num_states, start_state + worker_batch_size)
                GBFSWorker.queue(self.request_queue, end_state - start_state, backwards_range)

            # Wait for all of the workers to finish
            self.request_queue.join()

        self.quantity_buffer[self.quantity_buffer == 0] = 1

        num_attempts = self.quantity_buffer.astype(np.float32)
        solved_rate = self.solved_buffer / num_attempts
        average_cost = self.cost_buffer / num_attempts
        average_steps = self.step_buffer / self.solved_buffer.astype(np.float32)

        return solved_rate, average_steps, average_cost

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
