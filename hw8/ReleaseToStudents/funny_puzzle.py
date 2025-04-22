import heapq
import numpy as np
from copy import deepcopy

def state_check(state):
    """Validate state and return the corresponding goal state as a tuple."""
    non_zero = [n for n in state if n != 0]
    num_tiles = len(non_zero)
    if num_tiles == 0:
        raise ValueError('At least one number is not zero.')
    if num_tiles > 9:
        raise ValueError('At most nine numbers in the state.')
    if len(state) != 9 or not all(isinstance(n, int) for n in state):
        raise ValueError('State must be a list contain 9 integers.')
    if not all(0 <= n <= 9 for n in state):
        raise ValueError('The number in state must be within [0,9].')
    if len(set(non_zero)) != len(non_zero):
        raise ValueError('State can not have repeated numbers, except 0.')
    if sorted(non_zero) != list(range(1, num_tiles + 1)):
        raise ValueError('For puzzles with X tiles, the non-zero numbers must be within [1,X], '
                         'and there will be 9-X grids labeled as 0.')
    goal = list(range(1, num_tiles + 1)) + [0] * (9 - num_tiles)
    return tuple(goal)

def get_manhattan_distance(curr_state, goal_state):
    """Sum of Manhattan distances of each tile from its goal position."""
    # goal_state may be list or tuple
    goal_pos = {val: i for i, val in enumerate(goal_state) if val != 0}
    dist = 0
    for i, v in enumerate(curr_state):
        if v == 0:
            continue
        gi = goal_pos[v]
        x1, y1 = divmod(i, 3)
        x2, y2 = divmod(gi, 3)
        dist += abs(x1 - x2) + abs(y1 - y2)
    return dist

def naive_heuristic(curr_state, goal_state):
    """Trivial heuristic: always 0."""
    return 0

def sum_of_square_distances(curr_state, goal_state):
    """Sum of squared Euclidean distances of each tile from its goal."""
    goal_pos = {val: i for i, val in enumerate(goal_state) if val != 0}
    dist = 0
    for i, v in enumerate(curr_state):
        if v == 0:
            continue
        gi = goal_pos[v]
        x1, y1 = divmod(i, 3)
        x2, y2 = divmod(gi, 3)
        dist += (x1 - x2) ** 2 + (y1 - y2) ** 2
    return dist

def get_successors(state):
    """
    Return a sorted list of all valid successor states.
    Each successor is a Python list of 9 ints.
    """
    board = np.array(state).reshape(3, 3)
    zeros = list(zip(*np.where(board == 0)))
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    succs = set()

    for zr, zc in zeros:
        for dr, dc in moves:
            nr, nc = zr + dr, zc + dc
            if 0 <= nr < 3 and 0 <= nc < 3 and board[nr, nc] != 0:
                nb = deepcopy(board)
                nb[zr, zc], nb[nr, nc] = nb[nr, nc], nb[zr, zc]
                flat = [int(x) for x in nb.flatten()]
                succs.add(tuple(flat))

    # convert back to list of lists, sorted lexicographically
    return [list(t) for t in sorted(succs)]

def print_succ(state, heuristic=get_manhattan_distance):
    """Print each successor of state along with its h-value."""
    goal = state_check(state)
    for s in get_successors(state):
        h = heuristic(s, goal)
        print(s, f"h={h}")

def is_solvable(state):
    """Check solvability by counting inversions (ignoring zeros)."""
    nums = [n for n in state if n != 0]
    inv = sum(1 for i in range(len(nums)) for j in range(i+1, len(nums)) if nums[i] > nums[j])
    return inv % 2 == 0

def solve(state, heuristic=get_manhattan_distance):
    """
    A* search from state to its goal.
    Prints False if unsolvable, else prints True and the solution path:
      [state] h=X moves: Y
    followed by Max queue length.
    """
    goal = state_check(state)
    start = tuple(state)
    goal_t = tuple(goal)

    if start == goal_t:
        print(True)
        print(f"{state} h=0 moves: 0")
        print("Max queue length: 1")
        return

    if not is_solvable(state):
        print(False)
        return

    print(True)
    # Priority queue entries: (f = g+h, g, state_list, parent_index)
    pq = []
    heapq.heappush(pq, (heuristic(state, goal), 0, state, -1))

    # Best g found so far for each state
    best_g = {start: 0}
    # Visited with final g
    visited = {}

    # History for path reconstruction: (state, h, g, parent_index)
    history = []
    max_q = 1

    while pq:
        max_q = max(max_q, len(pq))
        f, g, curr, parent_idx = heapq.heappop(pq)
        curr_t = tuple(curr)

        # Skip if we've already found a better path
        if curr_t in visited and g >= visited[curr_t]:
            continue

        visited[curr_t] = g
        h = f - g
        idx = len(history)
        history.append((curr, h, g, parent_idx))

        if curr_t == goal_t:
            # Reconstruct path
            path = []
            i = idx
            while i != -1:
                st, hh, gg, pi = history[i]
                path.append((st, hh, gg))
                i = pi
            for st, hh, gg in reversed(path):
                print(f"{st} h={hh} moves: {gg}")
            print(f"Max queue length: {max_q}")
            return

        for nb in get_successors(curr):
            nb_t = tuple(nb)
            g2 = g + 1
            if nb_t not in best_g or g2 < best_g[nb_t]:
                best_g[nb_t] = g2
                h2 = heuristic(nb, goal)
                heapq.heappush(pq, (g2 + h2, g2, nb, idx))

        
