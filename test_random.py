from glob import glob
from time import time
from random import choice

from argparse import ArgumentParser

from sokoban import SokobanState, Astar
from sokoban.heuristics import GreedyHeuristic, ManhattanHeuristic, EuclidHeuristic, HungarianHeuristic, QLearningHeuristic

def action_to_string(action):
    if action == 0:
        return 'U'
    elif action == 1:
        return "R"
    elif action == 2:
        return "D"
    elif action == 3:
        return "L"

def main(walls: str, targets: int, steps: int):
    walls = glob(f"{walls}/*.txt")

    state = SokobanState.generate(choice(walls), num_targets=targets, num_steps=steps)

    heuristic = QLearningHeuristic("./qlearning_weights/convolution_network_1.torch")
    # heuristic = HungarianHeuristic("Manhattan")
    # heuristic = ManhattanHeuristic()

    print("Initial State")
    print("-" * 70)
    state.display()
    print("-" * 70)
    print()
    
    t0 = time()
    states, actions = Astar(state, heuristic)
    t1 = time()
    print("Solution Found!")
    print("-" * 70)
    print(f"Actions: {' '.join(map(action_to_string, actions))}")
    print(f"Time taken: {1000 * (t1 - t0)} ms.")

    input("Press enter to display the solution as a sequence of states.")
    for state in states:
        state.display()

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-w', "--walls", type=str, default='./walls', help="Directory with special wall files.")
    parser.add_argument('-t', "--targets", type=int, default=2, help="Number of targets to generate.")
    parser.add_argument('-s', "--steps", type=int, default=10_000, help="Number of steps to take when generating.")

    main(**parser.parse_args().__dict__)