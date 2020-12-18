# sokoban
Sokoban game AI for UCI CS271 course project

## Building the Environment

The Sokoban environment is implemented in C++ to ensure that the game can run quickly, 
and we can perform search efficiently.

In order to build the environment you will need:

- A modern C++ compiler capable of C++17 or above and OpenMP.
- Python 3.7 or above
- [Cython](https://pypi.org/project/Cython/)

Run `make` in the root directory in order to create the environment and the python wrapper.

## Python Instruction
In order to run the search algorithms, you will need the following libraries.

- Python 3.7 or above
- [Numpy](https://pypi.org/project/numpy/)
- [Scipy](https://pypi.org/project/scipy/)
- [fastdist](https://pypi.org/project/fastdist/)
- [PyTorch](https://pytorch.org/get-started/locally/)
- CUDA (Optional but makes DQN much faster)
- [TorchSpread](https://github.com/Alexanders101/TorchSpread) (Optional for training networks)

Run `source environment.sh` in the main directory in order to be able to use all of the python features from anywhere.

Run `python run.py -h` in order to get a help menu for the solver code. 


## Heuristic Options
Activate each of these using the `-s` options for `run.py`

1. Manhattan Greedy Selection
2. Euclidean Greedy Selection
3. Manhattan Hungarian Selection
4. Euclidean Hungarian Selection
5. Small Q-Learning Network (CPU)
6. Large Q-Learning Network (GPU & Batch A* Search)