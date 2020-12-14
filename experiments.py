from glob import glob
from time import time
from random import choice

from argparse import ArgumentParser

from sokoban import SokobanState, Astar
from sokoban.heuristics import GreedyHeuristic, ManhattanHeuristic, EuclidHeuristic, HungarianHeuristic

def action_to_string(action):
    if action == 0:
        return 'U'
    elif action == 1:
        return "R"
    elif action == 2:
        return "D"
    elif action == 3:
        return "L"

def main(wall: str, targets: int, steps: int):
    #walls = glob(f"{wall}/*.txt")

    state = SokobanState.generate(wall, num_targets=targets, num_steps=steps)

    heuristics = [      ManhattanHeuristic () ,             # greedy manhattan
                        EuclidHeuristic ()  ,               # greedy euclid
                        HungarianHeuristic ("Manhattan") ,  # hungarian manhattan
                        HungarianHeuristic ("Euclidean") ,  # hungarian euclid
    ]  

    times = [0,0,0,0]
    
    for i in range (4):
        t0 = time()
        states, actions = Astar(state, heuristics[i])
        t1 = time()

        times[i] = 1000 * (t1 - t0)

    print ("executed successfully!")

    for i in range (4):
        print (times[i])

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-w', "--wall", type=str, default='./walls/0000.txt', help="Directory with special wall files.")
    parser.add_argument('-t', "--targets", type=int, default=2, help="Number of targets to generate.")
    parser.add_argument('-s', "--steps", type=int, default=10_000, help="Number of steps to take when generating.")

    main(**parser.parse_args().__dict__)