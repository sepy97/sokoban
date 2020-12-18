from typing import Optional

import numpy as np

from sokoban.environment.sokoban_interface import SokobanState, parallel_expand, AstarData
from sokoban.heuristics import BaseHeuristic
from sokoban.search.astar import TPath


def BatchAstar(start: SokobanState, heuristic: BaseHeuristic, batch_size: int = 32, weight: float = 1.0) -> Optional[
    TPath]:
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
