from sokoban import SokobanState, Astar
from sokoban.heuristics import GreedyHeuristic

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
    heuristic = GreedyHeuristic()

    print("Initial State")
    print("-" * 70)
    state.display()
    print("-" * 70)
    print()
    
    states, actions = Astar(state, heuristic)
    print("Solution Found!")
    print("-" * 70)
    print(f"Actions: {' '.join(map(action_to_string, actions))}")
    # print(Astar(state, heuristic))