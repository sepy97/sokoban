from glob import glob
from time import time
from random import choice

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

def main(maps: str, setup: int, timing: int):
    #walls = glob(f"{wall}/*.txt")

    state = SokobanState.load (maps) #generate(wall, num_targets=targets, num_steps=steps)

    heuristics = [      ManhattanHeuristic () ,             # greedy manhattan
                        EuclidHeuristic ()  ,               # greedy euclid
                        HungarianHeuristic ("Manhattan") ,  # hungarian manhattan
                        HungarianHeuristic ("Euclidean") ,  # hungarian euclid
                        QLearningHeuristic ("./qlearning_weights/convolution_network_1.torch"), # QLearning
    ]  

    measured_time = 0
    
    t0 = time()
    states, actions = Astar (state, heuristics[setup-1])
    t1 = time()

    measured_time = 1000 * (t1 - t0)

    print (str (len (actions)), ' '.join(map(action_to_string, actions)))
    # str(actions))
    if timing == 1:
        print ("Execution time: ", measured_time)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-m', "--maps", type=str, default='./test_walls/sokoban01.txt', help="Directory with map files.")
    parser.add_argument('-s', "--setup", type=int, default=5, help="Heuristic to choose (from 1 to 5).")
    parser.add_argument('-t', "--timing", type=int, default=0, help="Timing is on (1) or off (0).")
    
    main(**parser.parse_args().__dict__)