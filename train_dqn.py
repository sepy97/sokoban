import sys
from typing import Optional, Tuple
from argparse import ArgumentParser

import numpy as np
import torch

from torch import optim
from tqdm import tqdm

from qlearning import DQNNetwork, TransitionCollector, GBFSEvaluator, Parameters, OutputManager, dqn_target, SokobanEnvironment
from torch_spread import NetworkManager, PlacementStrategy, DataParallelWrapper, TrainingWrapper


def load_parameters(parameters: Optional[str], override: bool) -> Tuple[Parameters, OutputManager]:
    hparams = Parameters.load(parameters)
    output_manager = OutputManager(hparams, reloaded=False)

    # Reload a previous run's hyper-parameters if they exist and we dont want to override them
    if output_manager.hparams_exist and not override:
        hparams = Parameters.load(output_manager.hparams_file)
        output_manager = OutputManager(hparams, reloaded=True)

    output_manager.save_hparams()
    return hparams, output_manager


def create_target_network(manager: NetworkManager, hparams: Parameters) -> DQNNetwork:
    target_network = DQNNetwork(False).to(manager.training_placement)
    target_network = manager.training_wrapper.wrap_network(target_network)
    target_network.load_state_dict(manager.state_dict)
    target_network.eval()
    return target_network


def create_optimizer(manager: NetworkManager, hparams: Parameters) -> optim.Optimizer:
    optimizer = None

    if 'apex' in hparams.training_optimizer:
        try:
            import apex.optimizers

            if hparams.training_optimizer == 'apex_adam':
                optimizer = apex.optimizers.FusedAdam

            elif hparams.training_optimizer == 'apex_lamb':
                optimizer = apex.optimizers.FusedLAMB

            else:
                optimizer = apex.optimizers.FusedSGD

        except ImportError:
            pass
    else:
        optimizer = getattr(optim, hparams.training_optimizer)

    if optimizer is None:
        print(f"Unable to load desired optimizer: {hparams.training_optimizer}.")
        print(f"Using Adam as a default.")
        optimizer = optim.Adam

    return optimizer(manager.training_parameters, lr=hparams.learning_rate)


def create_placement(hparams: Parameters):
    training_wrapper = TrainingWrapper()
    if torch.cuda.is_available() and hparams.use_gpu:
        placement = PlacementStrategy.round_robin_gpu_placement(hparams.num_networks)
        if torch.cuda.device_count() > 1:
            training_wrapper = DataParallelWrapper()
    else:
        placement = PlacementStrategy.uniform_cpu_placement(hparams.num_networks)

    return placement, training_wrapper


def create_training_iterator(epoch: int, hparams: Parameters):
    iterator = enumerate(range(0, hparams.training_states, hparams.training_batch_size))
    if hparams.progress_bar:
        iterator = tqdm(iterator,
                        desc=f"Epoch {epoch + 1}/{hparams.training_epochs}",
                        total=hparams.training_states // hparams.training_batch_size)

    return iterator


def numpy_to_torch_type(dtype):
    return {
        np.bool: torch.bool,
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128,
        np.dtype('uint8'): torch.uint8,
        np.dtype('int32'): torch.int32,
        np.dtype('bool'): torch.bool,
    }[dtype]


def main(parameters: Optional[str], generate: Optional[str], override: bool):
    # Create a default configuration json file
    if generate is not None:
        Parameters().save(generate)
        return

    hparams, output_manager = load_parameters(parameters, override)

    # Setup the logger file. Any print statements will be redirected to the file as well as display.
    sys.stdout = output_manager.logger
    output_manager.print_parameters()

    output_manager.print_with_border("Initializing")

    # Get the state information for the current environment
    env = SokobanEnvironment(48, "./walls", min_targets=1, max_targets=32)
    state_shapes, state_types = env.state_information()
    state_types = [numpy_to_torch_type(dtype) for dtype in state_types]

    # Device to place the worker and training networks
    placement, training_wrapper = create_placement(hparams)

    # Manages the networks and network workers
    manager = NetworkManager(input_shape=state_shapes,
                             input_type=state_types,
                             output_shape=env.num_actions_max,
                             output_type=None,
                             batch_size=hparams.worker_network_batch_size,
                             network_class=DQNNetwork,
                             network_args=[],
                             placement=placement,
                             training_wrapper=training_wrapper,
                             num_worker_buffers=2,
                             worker_amp=True)

    # Manages the parallel collection workers and replay buffer
    collector = TransitionCollector(hparams, manager.client_config)

    # Manages the parallel testing workers for network evaluation
    evaluator = GBFSEvaluator(hparams, manager.client_config)

    with manager, collector, evaluator:
        # Load any previous weights if they exist from a previous run
        initial_iteration = output_manager.load_weights(manager)

        # Create the slowly changing target network
        target_network = create_target_network(manager, hparams)

        # Create PyTorch optimizer
        optimizer = create_optimizer(manager, hparams)

        # Loop variables
        print()
        iteration = 0
        num_states = 0
        num_trained_states = 0

        while num_states < hparams.total_states:
            iteration += 1
            output_manager.print_with_border(f"Iteration {iteration}")

            # -------------------------------------------------------------------------------------
            # Simulation
            # -------------------------------------------------------------------------------------
            output_manager.print_with_border("Simulation")
            replay_buffer = collector.collect(hparams.simulation_states,
                                              hparams.n_step,
                                              hparams.back_max,
                                              True)

            num_states += hparams.simulation_states
            output_manager.print_value("Total Number of States", num_states)

            # Q-learning sometimes collects some warmup states before training
            if num_states < hparams.training_states:
                continue

            # The first iteration should be run with maximum randomness
            # Afterwards, switch to proper epsilon-greedy
            collector.update_epsilons(hparams.epsilon)

            # -------------------------------------------------------------------------------------
            # Training
            # -------------------------------------------------------------------------------------
            output_manager.print_with_border("Training")
            with manager.training_network as policy_network:
                for epoch in range(hparams.training_epochs):
                    average_loss = 0.0
                    for batch, state in create_training_iterator(epoch, hparams):
                        # Sample a batch from the replay buffer
                        current_batch_size = min(hparams.training_states - state, hparams.training_batch_size)
                        sample_index, sample, weights = replay_buffer.sample(current_batch_size)
                        num_trained_states += current_batch_size

                        # Move data to gpu if the training network is using it
                        sample = sample.to(manager.training_placement)
                        weights = weights.to(manager.training_placement)

                        # Calculate Bellman Q targets
                        targets = dqn_target(policy_network, target_network, sample, hparams, initial_iteration)

                        # Current estimates for the q network
                        policy_network.train()
                        states, actions = sample("states", raw=True), sample("actions")
                        all_q_values = policy_network(states)
                        q_values = all_q_values.gather(1, actions.unsqueeze(1)).squeeze()

                        # Compute bellman loss term, weighted by prioritized replay
                        # We keep the separate delta because it is used by the prioritized replay buffer
                        delta = targets - q_values
                        loss = delta * delta * weights
                        loss = loss.mean()

                        # Perform gradient descent step
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # Update persistent variables
                        replay_buffer.update_priority(sample_index, delta)
                        average_loss += loss.item()

                    average_loss = average_loss / (batch + 1)

                output_manager.print_value("Total Training States", num_trained_states, start='\n')
                output_manager.print_value("Latest Loss", average_loss)

                # if average_loss < hparams.loss_threshold:
                if (iteration % hparams.target_update_iterations) == 0:
                    print("Updating Target Network")
                    target_state_dict = target_network.state_dict()
                    value_state_dict = manager.state_dict

                    new_target_state_dict = {}
                    for key in target_state_dict:
                        new_target_state_dict[key] = (target_state_dict[key] * hparams.alpha +
                                                      value_state_dict[key] * (1 - hparams.alpha))

                    target_network.load_state_dict(new_target_state_dict)

                output_manager.save_weights(manager, iteration)
                replay_buffer.update_max_priority()
                initial_iteration = False

            # -------------------------------------------------------------------------------------
            # Testing
            # -------------------------------------------------------------------------------------
            if (iteration % hparams.testing_iterations) == 0:
                output_manager.print_with_border(f"Evaluation")
                # Perform an evaluation run
                test_results = evaluator.evaluate(hparams.testing_states, hparams.testing_max_steps)
                display_indices = np.round(np.linspace(0, hparams.testing_max_steps, 10)).astype(np.int64)
                display_indices = np.unique(display_indices)

                test_results = [result[display_indices] for result in test_results]
                for distance, solved, steps, costs in zip(display_indices, *test_results):
                    print(f"Solved {100 * solved:.2f}% of states of distance {distance}.", end=' ')
                    print(f"Average steps taken: {steps:.2f}.", end=' ')
                    print(f"Average cost to go: {costs:.2f}")
                print()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--parameters", default=None, type=str,
                        help="A link to a json file storing parameters to be parsed")

    parser.add_argument('-o', '--override', action='store_true',
                        help="Override saved hparams with new ones if present.")

    parser.add_argument('-g', '--generate', default=None, type=str,
                        help="Generate json file to destination from default parameters.")

    main(**parser.parse_args().__dict__)
