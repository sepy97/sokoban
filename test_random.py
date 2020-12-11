from glob import glob
from time import time
from random import choice

from sokoban import SokobanState, Astar
from sokoban.heuristics import GreedyHeuristic, ManhattanHeuristic, EuclidHeuristic

def action_to_string(action):
    if action == 0:
        return 'U'
    elif action == 1:
        return "R"
    elif action == 2:
        return "D"
    elif action == 3:
        return "L"

def main():
    walls = glob("./walls/*.txt")

    state = SokobanState.generate(choice(walls), num_targets=2, num_steps=10_000)
    heuristic = ManhattanHeuristic()

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
    main()