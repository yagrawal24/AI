import numpy as np
import heapq
import math
import copy


def print_succ(state):
    lists = []
    h = []
    lists, h = help_print_succ(state)
    for index in range(len(lists)):
        print("{} h={}".format(lists[index], h[index]))


def manhattan(puzzle):
    sum = 0
    for i in range(8):
        # Get index
        index = puzzle.index(i + 1)
        # Calculate manhattan distance
        row = math.floor(index / 3)
        col = index % 3

        expected_row = math.floor(i / 3)
        expected_col = i % 3

        dist = abs(expected_row - row) + abs(expected_col - col)
        # print("Number: {}, row: {}, col: {}, expectedrow: {}, expected_col: {}, dist: {}".format(index, row, col, expected_row, expected_col, dist))
        sum += dist

    return sum


def help_print_succ(state):
    lists = []
    h = []
    # Locate 0 index
    index = state.index(0)
    # Find possible movements
    row = math.floor(index / 3)
    col = index % 3

    right = False
    left = False
    up = False
    down = False

    if col + 1 < 3:
        right = True
    if col - 1 >= 0:
        left = True
    if row - 1 >= 0:
        up = True
    if row + 1 < 3:
        down = True

    # Do movements and add to list
    if right:
        temp = copy.deepcopy(state)
        temp[index] = temp[index + 1]
        temp[index + 1] = 0
        lists.append(temp)

    if left:
        temp = copy.deepcopy(state)
        temp[index] = temp[index - 1]
        temp[index - 1] = 0
        lists.append(temp)
    if up:
        temp = copy.deepcopy(state)
        temp[index] = temp[index - 3]
        temp[index - 3] = 0
        lists.append(temp)
    if down:
        temp = copy.deepcopy(state)
        temp[index] = temp[index + 3]
        temp[index + 3] = 0
        lists.append(temp)

    lists = sorted(lists)
    for lstate in lists:
        h.append(manhattan(lstate))

    return lists, h


def solve(state):
    open = []
    closed = []
    visited = []
    g = 0
    h = manhattan(state)
    heapq.heappush(open, (g + h, state, (g, h, -1)))
    visited.append(state)
    temp = None
    while len(open):
        temp = heapq.heappop(open)
        heapq.heappush(closed, temp)
        if temp[2][1] == 0:
            break
        else:
            lists, h = help_print_succ(temp[1])
            for lstate in lists:
                if lstate not in visited:
                    visited.append(lstate)
                    heapq.heappush(open, (manhattan(lstate) + temp[2][0] + 1, lstate, (temp[2][0] + 1, manhattan(lstate), len(closed) - 1)))
    solution = [temp]
    index = temp[2][2]
    while index != -1:
        solution.append(closed[index])
        index = closed[index][2][2]

    for i in range(len(solution)):
        print("{} h={} moves: {}".format(solution[len(solution) - i - 1][1], solution[len(solution) - i - 1][2][1], solution[len(solution) - i - 1][2][0]))
