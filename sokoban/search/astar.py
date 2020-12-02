from queue import PriorityQueue
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple, List, Optional

from sokoban.environment.sokoban_interface import SokobanState, load_state, expand_state, next_state
from sokoban.heuristics import BaseHeuristic

TPath = Tuple[List[SokobanState], List[int]]

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)
        
def Astar_path(start: SokobanState, 
               goal: SokobanState, 
               parents: Mapping[SokobanState, Tuple[SokobanState, int]]) -> TPath:
    path = []
    actions = []
    
    state = goal
    while state != start:
        path.append(state)
        state, action = parents[state]
        actions.append(action)
    
    path.append(start)
    path.reverse()
    actions.reverse()
    return path, actions

def Astar(start: SokobanState, heuristic: BaseHeuristic) -> Optional[TPath]:
    # Initialize search data structures
    parents: Mapping[SokobanState, Tuple[SokobanState, int]] = defaultdict(lambda: None)
    costs: Mapping[SokobanState, float] = defaultdict(lambda: float('inf'))
    open_set: PriorityQueue[PrioritizedItem] = PriorityQueue()

    # Add the start state at the from of the queue and set its path to 0.
    open_set.put(PrioritizedItem(0, start))
    costs[start] = 0
    
    while not open_set.empty():
        state = open_set.get().item
        state_cost = costs[state]
        
        if state.solved:
            return Astar_path(start, state, parents)
        
        children = expand_state(state)
        for action, child in enumerate(children):
            cost = state_cost + 1
            if cost < costs[child]:
                parents[child] = (state, action)
                costs[child] = cost
                open_set.put(PrioritizedItem(cost + heuristic(child), child))
    
    return None