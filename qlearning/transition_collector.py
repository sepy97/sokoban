from typing import Tuple, List, Union, Optional, Dict

import numpy as np
import torch
import os
from torch_spread import mp_ctx

from qlearning.parameters import Parameters
from qlearning.qlearning import OrderedReplayBuffer, EpsilonGreedyClient
from qlearning.output_manager import StatusBarUpdater, AsyncStatusBar
from qlearning.sokoban_environment import SokobanEnvironment


class TransitionWorker(mp_ctx.Process):
    KILL = b'KILL'
    UPDATE = b'UPDATE'
    COLLECT = b'COLLECT'

    def __init__(self,
                 hparams: Parameters,
                 client_config: dict,
                 progress_bar: StatusBarUpdater,
                 replay_buffer: OrderedReplayBuffer,
                 request_queue: mp_ctx.JoinableQueue):
        super(TransitionWorker, self).__init__()

        self.hparams = hparams
        self.client_config = client_config
        self.request_queue = request_queue
        self.replay_buffer = replay_buffer
        self.progress_bar = progress_bar

        self.ready_event = mp_ctx.Event()

    @staticmethod
    def state_vectors(env: SokobanEnvironment, states) -> List[torch.Tensor]:
        """ Helper function to convert state objects into PyTorch tensors. """
        vectors = [torch.from_numpy(v) for v in env.state_to_nnet_input(states)]
        return vectors

    @staticmethod
    def queue(request_queue, n_step: int, batch_size, backwards_range: Tuple[int, int]):
        """ Queue up dynamically generated transition collection. """
        request_queue.put((TransitionWorker.COLLECT, n_step, batch_size, backwards_range))

    @staticmethod
    def queue_from_file(request_queue, n_step: int, batch_size, load_directory: str):
        """ Queue up loaded transitions from a file. """
        request_queue.put((TransitionWorker.COLLECT, n_step, batch_size, load_directory))

    def shutdown(self):
        self.request_queue.put((self.KILL,))

    def update_epsilon(self, epsilon: float):
        self.request_queue.put((self.UPDATE, epsilon))

    def collect_rollout(self,
                        env: SokobanEnvironment,
                        client: EpsilonGreedyClient,
                        batch_size: int,
                        n_step: int,
                        backwards_range: Union[Tuple[int, int], List[int], str]) -> Dict[str, torch.Tensor]:

        initial_states, distances = env.generate_states(batch_size, backwards_range)

        # We train on the first actions that we take from the starting state
        initial_actions = np.zeros(batch_size, dtype=np.int64)

        # Keep track of rolling terminals along this episode
        terminals = env.is_solved(initial_states)

        # Compute rolling discount cost as we go
        discounts = np.ones(batch_size, dtype=np.float32)
        discount_costs = np.zeros(batch_size, dtype=np.float32)

        # Set up loop variables
        final_states = initial_states

        # Perform N-Step rollout
        for step in range(n_step):
            # Sample the neural network for an action
            # If this is our first action, then store it as the training action.
            actions = client.sample_actions_with_batching(self.state_vectors(env, final_states))
            if step == 0:
                initial_actions[:] = actions

            # Perform a simulation step
            final_states, costs = env.next_state(final_states, actions)

            # Compute rolling discount costs
            # Note that for cost-to-go formulation:
            # We set to cost to 0 if we've ever encountered the solved state prior to this.
            # This is why we compute the cost before updating the terminals.
            discount_costs += discounts * (1 - terminals) * np.asarray(costs)

            # Update rolling terminals to see if any of the transitions has ended
            terminals |= env.is_solved(final_states)

            # Update rolling discounts
            discounts *= self.hparams.discount

        return {
            "states": self.state_vectors(env, initial_states),
            "results": self.state_vectors(env, final_states),
            "distances": torch.as_tensor(distances, dtype=torch.float32),
            "actions": torch.as_tensor(initial_actions, dtype=torch.int64),
            "terminals": torch.as_tensor(terminals, dtype=torch.uint8),
            "priorities": torch.zeros(batch_size, dtype=torch.float32),
            "discounts": torch.as_tensor(discounts, dtype=torch.float32),
            "discount_costs": torch.as_tensor(discount_costs, dtype=torch.float32)
        }

    def run(self):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

        # Set random seed to ensure workers perform different actions
        if hasattr(self, "pid"):
            torch.manual_seed(self.pid)
            np.random.seed(self.pid)

        # Set up worker variables
        env = SokobanEnvironment(48, "./walls", min_targets=1, max_targets=32)
        simulation_batch_size = self.hparams.simulation_batch_size
        network_batch_size = self.hparams.simulation_network_batch_size
        client = EpsilonGreedyClient

        # Begin the simulation process with a fully random epsilon
        # We later set the epsilon during training
        with client(self.client_config, network_batch_size, 1.0) as client:
            self.ready_event.set()

            while True:
                # Parse Command
                command, *data = self.request_queue.get()

                # Option 1: Kill switch
                if command == self.KILL:
                    self.request_queue.task_done()
                    return

                # Option 2: Update client epsilon
                elif command == self.UPDATE:
                    client.epsilon = data[0]
                    self.request_queue.task_done()
                    continue

                # Option 3: Perform a simulation
                with env:
                    n_step, num_states, backwards_range = data

                    for state_number in range(0, num_states, simulation_batch_size):
                        # Current batch size is what is left of the task bounded by batch size
                        current_batch_size = min(simulation_batch_size, num_states - state_number)

                        # Collect monte carlo q-learning rollout for n steps
                        rollout = self.collect_rollout(env, client, current_batch_size, n_step, backwards_range)

                        # Add the previous rollout to the global replay buffer
                        self.replay_buffer.put(rollout, current_batch_size)

                        # Inform the progress bar that this batch is finished
                        self.progress_bar.update(current_batch_size)

                    self.request_queue.task_done()


class TransitionCollector:
    def __init__(self, hparams: Parameters, client_config: dict):
        self.hparams = hparams
        self.client_config = client_config

        self.request_queue = mp_ctx.JoinableQueue()
        self.workers: Optional[List[TransitionWorker]] = None
        self.progress_bar = AsyncStatusBar(enable=self.hparams.progress_bar)

        buffer_class = OrderedReplayBuffer
        self.replay_buffer = buffer_class(client_config["input_shape"],
                                          client_config["input_type"],
                                          hparams.replay_buffer_size)

    def start(self, wait_for_start: bool = True):
        if self.workers is not None:
            raise AssertionError("Cannot start the collector twice.")

        print("Starting Collector")
        print("Creating shared memory buffer. This might take a while...")
        worker_arguments = [self.hparams,
                            self.client_config,
                            self.progress_bar.updater,
                            self.replay_buffer,
                            self.request_queue]

        self.workers = []
        for i in range(self.hparams.simulation_workers):
            print("-", end='')
            if ((i + 1) % 60) == 0:
                print()

            worker = TransitionWorker(*worker_arguments)
            worker.start()
            self.workers.append(worker)

        if wait_for_start:
            for worker in self.workers:
                worker.ready_event.wait()

        print("\nCollector Ready")

    def stop(self):
        if self.workers is None:
            raise AssertionError("Cannot stop a collector that isn't started")

        for worker in self.workers:
            worker.shutdown()

        for worker in self.workers:
            worker.join(timeout=1)

        self.request_queue.join()
        self.workers = None

    def collect(self,
                num_states: int,
                n_step: int,
                backwards_range: Union[int, Tuple[int, int], List[int], str],
                clear_buffer: bool = False):
        if self.workers is None:
            raise AssertionError("Cannot collect until collector is started.")

        # Setup input: We can either load from a file or generate fresh states
        if isinstance(backwards_range, str):
            queue_method = TransitionWorker.queue_from_file
        else:
            queue_method = TransitionWorker.queue
            if not isinstance(backwards_range, (list, tuple)):
                backwards_range = (0, backwards_range)

        # Optionally clear the buffer
        if clear_buffer:
            self.replay_buffer.reset()

        with self.progress_bar("Collecting", num_states):
            # Queue up all of the batches
            worker_batch_size = num_states // len(self.workers)
            for start_state in range(0, num_states, worker_batch_size):
                end_state = min(num_states, start_state + worker_batch_size)
                current_batch_size = end_state - start_state
                queue_method(self.request_queue, n_step, current_batch_size, backwards_range)

            # Wait for all of the workers to finish
            self.request_queue.join()

        return self.replay_buffer

    def update_epsilons(self, epsilon):
        if self.workers is None:
            raise AssertionError("Not Started.")

        for worker in self.workers:
            worker.update_epsilon(epsilon)

        self.request_queue.join()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
