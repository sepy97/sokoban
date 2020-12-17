from sokoban import SokobanState, Astar
from sokoban.heuristics import HungarianHeuristic
from time import time

def action_to_string(action):
    if action == 0:
        return 'U'
    elif action == 1:
        return "R"
    elif action == 2:
        return "D"
    elif action == 3:
        return "L"

if __name__ == "__main__":
    state = SokobanState.load("./sokoban01.txt")
    
    #heuristic = HungarianHeuristic('Manhattan')
    heuristic = HungarianHeuristic('Euclidean')

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
    # print(Astar(state, heuristic))
