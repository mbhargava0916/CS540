from funny_puzzle import *

# Test the example input
solve([1, 2, 3, 4, 5, 6, 0, 7, 0], heuristic=naive_heuristic)
solve([1, 2, 3, 4, 5, 6, 0, 0, 7], heuristic=get_manhattan_distance)
solve([4, 3, 0, 5, 1, 6, 7, 2, 0], heuristic=get_manhattan_distance)
solve([1, 0, 0, 0, 2, 3, 0, 4, 5], heuristic=get_manhattan_distance)