from queue import PriorityQueue
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple, List, Optional, Dict
from heapq import heappop, heappush
from itertools import chain

import numpy as np
from sokoban.environment.sokoban_interface import SokobanState, parallel_expand, AstarData
from sokoban.heuristics import BaseHeuristic
from sokoban.search.astar import Astar_path, PrioritizedItem, TPath
        

# class AstarData:
#     def __init__(self, initial_size = 1024):
#         self.current_length = 0
#         self.current_size = initial_size
#
#         self.parents = np.full(initial_size, -1, dtype=np.int64)
#         self.actions = np.full(initial_size, -1, dtype=np.int64)
#         self.costs = np.full(initial_size, np.inf, dtype=np.float32)
#         self.state_table = np.full(initial_size, np.iinfo(np.int64).max, dtype=np.int64)
#
#         self.open_set: List[Tuple[float, int]] = []
#
#         self.state_mapping: Dict[SokobanState, int] = {}
#         self.inverse_state_mapping: Dict[int, SokobanState] = {}
#
#     def __len__(self):
#         return self.current_length
#
#     def add(self, state: SokobanState):
#         if state in self.state_mapping:
#             return
#
#         if self.current_length >= self.current_size:
#             self.current_size = self.current_size * 2
#
#             new_parents = np.full(self.current_size, -1, dtype=np.int64)
#             new_actions = np.full(self.current_size, -1, dtype=np.int64)
#             new_costs = np.full(self.current_size, np.inf, dtype=np.float32)
#             new_state_table = np.full(self.current_size, np.iinfo(np.int64).max, dtype=np.int64)
#
#             new_costs[:self.current_length] = self.costs[:]
#             new_parents[:self.current_length] = self.parents[:]
#             new_actions[:self.current_length] = self.actions[:]
#             new_state_table[:self.current_length] = self.state_table[:]
#
#             self.costs = new_costs
#             self.parents = new_parents
#             self.actions = new_actions
#             self.state_table = new_state_table
#
#         self.state_mapping[state] = self.current_length
#         self.inverse_state_mapping[self.current_length] = state
#
#         self.current_length += 1
#
#     def push(self, state: SokobanState, priority: int):
#         self.add(state)
#         heappush(self.open_set, (priority, self.state_to_id(state)))
#
#     def pop(self) -> SokobanState:
#         return self.id_to_state(heappop(self.open_set)[1])
#
#     def push_id(self, state_id: int, priority: int):
#         heappush(self.open_set, (priority, state_id))
#
#     def pop_id(self) -> int:
#         return heappop(self.open_set)[1]
#
#     def open_empty(self):
#         return len(self.open_set) == 0
#
#     def state_to_id(self, state: SokobanState) -> int:
#         return self.state_mapping[state]
#
#     def id_to_state(self, id: int) -> SokobanState:
#         return self.inverse_state_mapping[id]
#
#     def path(self, start: SokobanState, goal: SokobanState):
#         path = []
#         actions = []
#
#         start_id = self.state_to_id(start)
#         state_id = self.state_to_id(goal)
#
#         while state_id != start_id:
#             path.append(self.id_to_state(state_id))
#             action = self.actions[state_id]
#             state_id = self.parents[state_id]
#             actions.append(action)
#
#         path.append(start)
#         path.reverse()
#         actions.reverse()
#         return path, actions


def BatchAstar(start: SokobanState, heuristic: BaseHeuristic, batch_size: int = 32, weight: float = 1.0) -> Optional[TPath]:
    initial_size = start.size_x * start.size_y * start.targets.shape[0] * 128
    data = AstarData(initial_size)

    # Add the start state at the from of the queue and set its path to 0.
    data.push(start, 0)
    data.set_cost(0, 0)
    num_of_moves = 0
    
    while data.has_elements():
        num_of_moves += 1

        states, state_ids = data.pop_batch(batch_size)

        # Check to see if any of our future states are the solved state and jump out
        solved_state = data.check_solved(states)
        if solved_state:
            print(f"States explored: {len(data)}")
            return data.extract_path(start, solved_state)

        # Expand the current batch of states and add them to the buffers
        children = parallel_expand(states)
        heuristic_values = np.asarray(heuristic.batch_call(children), dtype=np.float32)

        data.process_children(
            state_ids,
            heuristic_values,
            children,

            num_of_moves,
            weight
        )

    return None
