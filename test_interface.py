from time import time
from tqdm import tqdm

from sokoban import SokobanState

N = 10_000_000

if __name__ == "__main__":
    state = SokobanState.load("./sokoban00.txt")

    print("Original State")
    print("-" * 70)
    state.display()
    print("-" * 70)
    print()
    
    result = state.next_state(2)
    print("New State")
    print("-" * 70)
    result.display()
    print("-" * 70)
    print()

    print("Original State Again. Ensure Immutable.")
    print("-" * 70)
    state.display()
    print("-" * 70)
    print()
    
    print("Running timing test:")
    print("Watch your memory on this! If we dont have a leak then this should not use any ram.")
    print("-" * 70)

    t0 = time()
    for _ in tqdm(range(N)):
        state.next_state(2)
    t1 = time()
    print(f"Average time per call: {1000 * 1000 * 1000 * (t1 - t0) / N:.3f} ns")

    t0 = time()
    for _ in tqdm(range(N)):
        state.expand()
    t1 = time()
    print(f"Average time per expand: {1000 * 1000 * 1000 * (t1 - t0) / N:.3f} ns")

    t0 = time()
    for _ in tqdm(range(N)):
        state.dead_lock
    t1 = time()
    print(f"Average time per deadlock: {1000 * 1000 * 1000 * (t1 - t0) / N:.3f} ns")
 