import heapq

def state_check(state):
    """check the format of state, and return corresponding goal state.
       Do NOT edit this function."""
    non_zero_numbers = [n for n in state if n != 0]
    num_tiles = len(non_zero_numbers)
    if num_tiles == 0:
        raise ValueError('At least one number is not zero.')
    elif num_tiles > 9:
        raise ValueError('At most nine numbers in the state.')
    matched_seq = list(range(1, num_tiles + 1))
    if len(state) != 9 or not all(isinstance(n, int) for n in state):
        raise ValueError('State must be a list contain 9 integers.')
    elif not all(0 <= n <= 9 for n in state):
        raise ValueError('The number in state must be within [0,9].')
    elif len(set(non_zero_numbers)) != len(non_zero_numbers):
        raise ValueError('State can not have repeated numbers, except 0.')
    elif sorted(non_zero_numbers) != matched_seq:
        raise ValueError('For puzzles with X tiles, the non-zero numbers must be within [1,X], '
                          'and there will be 9-X grids labeled as 0.')
    goal_state = matched_seq
    for _ in range(9 - num_tiles):
        goal_state.append(0)
    return tuple(goal_state)

def get_manhattan_distance(from_state, to_state):
    distance = 0
    for num in range(1, 10):  # skip 0s
        if num not in from_state or num not in to_state:
            continue
        from_idx = from_state.index(num)
        to_idx = to_state.index(num)
        fx, fy = divmod(from_idx, 3)
        tx, ty = divmod(to_idx, 3)
        distance += abs(fx - tx) + abs(fy - ty)
    return distance

def naive_heuristic(from_state, to_state):
    return 0

def sum_of_squares_distance(from_state, to_state):
    distance = 0
    for num in range(1, 10):
        if num not in from_state or num not in to_state:
            continue
        from_idx = from_state.index(num)
        to_idx = to_state.index(num)
        fx, fy = divmod(from_idx, 3)
        tx, ty = divmod(to_idx, 3)
        distance += (fx - tx) ** 2 + (fy - ty) ** 2
    return distance

def get_succ(state):
    succ_states = []
    for idx, val in enumerate(state):
        if val != 0:
            continue
        x, y = divmod(idx, 3)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                n_idx = 3 * nx + ny
                if state[n_idx] != 0:
                    new_state = list(state)
                    new_state[idx], new_state[n_idx] = new_state[n_idx], new_state[idx]
                    succ_states.append(new_state)
    return sorted(succ_states)

def print_succ(state, heuristic=get_manhattan_distance):
    goal_state = state_check(state)
    succ_states = get_succ(state)
    for succ_state in succ_states:
        print(succ_state, "h={}".format(heuristic(succ_state, goal_state)))

def is_solvable(state):
    non_zero = [n for n in state if n != 0]
    inv_count = 0
    for i in range(len(non_zero)):
        for j in range(i + 1, len(non_zero)):
            if non_zero[i] > non_zero[j]:
                inv_count += 1
    return inv_count % 2 == 0

def solve(state, heuristic=get_manhattan_distance):
    goal_state = state_check(state)
    if not is_solvable(state):
        print(False)
        return
    print(True)

    visited = set()
    pq = []
    heapq.heappush(pq, (heuristic(state, goal_state), state, (0, heuristic(state, goal_state), -1)))
    history = []
    max_length = 0

    while pq:
        max_length = max(max_length, len(pq))
        cost, curr_state, (g, h, parent_index) = heapq.heappop(pq)
        curr_tuple = tuple(curr_state)

        if curr_tuple in visited:
            continue
        visited.add(curr_tuple)

        this_index = len(history)
        history.append((curr_state, h, g, parent_index))

        if curr_state == list(goal_state):
            path = []
            i = this_index
            while i != -1:
                cs, h_val, moves, parent = history[i]
                path.append((cs, h_val, moves))
                i = parent
            for p in reversed(path):
                print(f"{p[0]} h={p[1]} moves: {p[2]}")
            print(f"Max queue length: {max_length}")
            return

        for succ in get_succ(curr_state):
            succ_tuple = tuple(succ)
            if succ_tuple not in visited:
                g_new = g + 1
                h_new = heuristic(succ, goal_state)
                heapq.heappush(pq, (g_new + h_new, succ, (g_new, h_new, this_index)))

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    # print_succ([2,5,1,4,0,6,7,0,3])
    # print()
    #
    # print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    # print()

    solve([2,5,1,4,0,6,7,0,3], heuristic=get_manhattan_distance)
    print()
