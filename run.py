import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from time import time
from argparse import ArgumentParser

import torch

from sokoban import SokobanState, Astar, BatchAstar
from sokoban.heuristics import ManhattanHeuristic, EuclidHeuristic, HungarianHeuristic, QLearningHeuristic


def action_to_string(action):
    if action == 0:
        return 'U'
    elif action == 1:
        return "R"
    elif action == 2:
        return "D"
    elif action == 3:
        return "L"


def main(map: str, setup: int):
    if torch.cuda.is_available():
        print("Cuda found. Running DQN network on GPU.")
        cuda = True
    else:
        print("Cuda not found. Running DQN network on CPU. This will be much slower.")
        cuda = False

    state = SokobanState.load(map)

    heuristics = [ ("Manhattan Greedy", ManhattanHeuristic()),
                   ("Euclidean Greedy", EuclidHeuristic ()),
                   ("Manhattan Hungarian", HungarianHeuristic("Manhattan")),
                   ("Euclidean Hungarian", HungarianHeuristic("Euclidean")),

                   ("Small Q-Learning", QLearningHeuristic ("./qlearning_weights/convolution_network_3.torch",
                                                            max_size=32,
                                                            cuda=False,
                                                            full_input=False)),

                   ("Large Q-Learning", QLearningHeuristic("./qlearning_weights/convolution_network_5.torch",
                                                           max_size=48,
                                                           full_input=True,
                                                           cuda=cuda))
    ]


    print("=" * 80)
    print(f"Running Sokoban file: {map}")
    print("-" * 80)
    state.display()
    print("-" * 80)

    name, heuristic = heuristics[setup - 1]

    print()
    print("-" * 80)
    print(f"Using Heuristic: {name}")
    print("-" * 80)
    t0 = time()

    if "Largest Q-Learning" in name:
        print("Running Batch A* Search")
        states, actions = BatchAstar(state, heuristic, batch_size=512, weight=10)
    else:
        print("Running A* Search")
        states, actions = Astar(state, heuristic)

    t1 = time()
    print(f"Runtime: {t1 - t0} seconds")
    print(f"Solution Length {len(actions)}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-m', "--map", type=str, default='./sokoban_benchmarks/sokoban01.txt', help="Map file to test on.")
    parser.add_argument('-s', "--setup", type=int, default=5, help="Heuristic to choose (from 1 to 6).")

    main(**parser.parse_args().__dict__)