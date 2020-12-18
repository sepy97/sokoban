import json
from typing import Optional
from argparse import Namespace


class Parameters(Namespace):
    def __init__(self):
        super(Parameters, self).__init__()

        # Directory to place output weights, logfile, and parameters
        self.output_directory = "./results/sokoban"

        # Add a timestamp to the output directory
        # This will disable automatic reloading of parameters and weights
        self.append_time_to_output = True

        # Total number of states to train on.
        self.total_states: int = 100_000_000_000

        # Whether or not we will allow networks to be placed on the gpu.
        # This will automatically place on all visible gpus.
        self.use_gpu: bool = True

        # Number of network workers to launch. Should typically be the number of gpus.
        self.num_networks: int = 1

        # -----------------------------------------------------------------------------------------
        # Simulation Options
        # =========================================================================================

        # Maximum number of backwards steps to take when generating states.
        self.back_max = 10000

        # Number of workers to launch for simulation collection.
        # Should be between num_cpu and 2 * num_cpu.
        # It is helpful to over-schedule workers so that some may be predicting
        # on the GPU while others are computing states on the CPU.
        # Be careful of creating too many processes and hitting the system limit!
        self.simulation_workers: int = 16

        # How many states a single collection worker will simulate at once
        self.simulation_batch_size: int = 512

        # How many states a single collection worker will send to the networks at once
        self.simulation_network_batch_size: int = 512

        # Total Number of states to process during collection.
        # Workers will collected 'simulation_batch_size' states at a time.
        # Enqueued sequentially.
        self.simulation_states: int = 16 * self.simulation_batch_size * self.simulation_workers

        # Total batch size for each worker network
        # Default is optimized to allocate networks evenly
        self.worker_network_batch_size = self.simulation_network_batch_size * self.simulation_workers
        self.worker_network_batch_size //= self.num_networks
        self.worker_network_batch_size += 1

        # -----------------------------------------------------------------------------------------
        # Training Options
        # =========================================================================================

        # Batch size during network training.
        # If using multiple gpus, then this will be spread across the gpus.
        self.training_batch_size: int = 512

        # Number of states to train on after collection.
        # If clearing the buffer, then this should probably be the number of collected states.
        self.training_states: int = self.simulation_states

        # Number of times to train over the previously collected sample
        self.training_epochs: int = 1

        # PyTorch optimizer to use for training
        # Must be accessible from torch.optim
        self.training_optimizer: str = "apex_adam"

        # Optimizer learning rate
        self.learning_rate: float = 0.0001

        # How many iterations to wait before updating target network
        self.target_update_iterations: int = 1

        # Create backup weights files every X training iterations
        self.weights_save_iterations: int = 5

        # -----------------------------------------------------------------------------------------
        # Testing Options
        # =========================================================================================

        # Number of parallel workers to launch
        self.testing_workers: int = 16

        # Maximum generation distance during training.
        self.testing_max_steps: int = 100

        # Maximum number of actions to perform during BFS before exiting.
        self.testing_max_actions: int = 2 * self.testing_max_steps

        # Number of training iterations to wait before testing.
        self.testing_iterations: int = 1

        # Total number of states to test
        self.testing_states: int = self.simulation_network_batch_size * self.testing_workers

        # -----------------------------------------------------------------------------------------
        # Q-Learning Options
        # =========================================================================================

        # Epsilon for epsilon-greedy policy. Initial iteration will still be fully random.
        # If using a boltzmann policy, then temperature = 1 / (1 - epsilon)
        self.epsilon: float = 0.2

        # Discount factor gamma for discount rewards. A value of 1.0 turns off discounts.
        self.discount: float = 1.0

        # Weight for exponential weighted target network.
        # Set to 0.0 to have classical Q-learning updated
        self.alpha: float = 0.2

        # Total size of the replay buffer.
        # Acts like a ring buffer, overflowing will start overwriting older data.
        self.replay_buffer_size: int = self.simulation_states

        # Number of steps to perform during a rollout. 1 step is classical q learning. n-step is TD(n).
        self.n_step: int = 3

        # Whether or not to display a progress bar during simulation and training
        self.progress_bar: bool = True

    def save(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.__dict__, file, separators=(',\n', ': '))

    @classmethod
    def load(cls, filepath: Optional[str] = None):
        result = cls()

        if filepath:
            with open(filepath, 'r') as file:
                for key, value in json.load(file).items():
                    setattr(result, key, value)

        return result
