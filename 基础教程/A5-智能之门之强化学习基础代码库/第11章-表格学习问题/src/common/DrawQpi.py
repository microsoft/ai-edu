
import numpy as np
from enum import Enum

# 注意这里的顺序和前面的章节不一样
LEFT, DOWN, RIGHT, UP = 3, 2, 1, 0

dict_arrows = {
    "[3]":   "←− ",
    "[2]":   " ↓ ",
    "[1]":   " −→",
    "[0]":   " ↑ ",
    "[2, 3]": "─┐ ",
    "[1, 2]": " ┌─",
    "[0, 1]": " └─",
    "[0, 3]": "─┘ ",
    "[1, 3]": "←−→",
    "[0, 2]": " ↕ ",
    "[1, 2, 3]": "─┬─",
    "[0, 1, 2]": " ├─",
    "[0, 1, 3]": "─┴─",
    "[0, 2, 3]": "─┤ ",
    "[0, 1, 2, 3]": "─┼─"
}


class CellType(Enum):
    common = 2
    start = 0
    goal = 1
    end = -1

class GridCell(object):
    def __init__(self, q, round: int, state_type: int):
        if state_type == CellType.common:  # common state
            self.q = np.round(q, round)
            if np.all(self.q == 0):
                self.space = " · "
            else:
                best_actions = np.argwhere(self.q == np.max(self.q))
                pos = best_actions.flatten().tolist()
                pos_s = str(pos)
                self.space = dict_arrows[pos_s]
        # elif state_type == CellType.start:  # start state
        #     self.space = "  S  "
        elif state_type == CellType.goal:  # goal state
            self.space = " G "
        elif state_type == CellType.end:  # end state
            self.space = " X "
        else:
            raise Exception("未知的状态类型")           


class Grid(object):
    def __init__(self, Q, shape: tuple, round: int, start_state: int, goal_state: int, end_state: list):
        self.array = np.empty(shape, dtype="<U3")
        for i in range(len(Q)):
            row = (int)(i / shape[1])
            col = (int)(i % shape[1])
            q = Q[i]
            if i == start_state:
                cell = GridCell(q, round, CellType.start)
            elif i == goal_state:
                cell = GridCell(q, round, CellType.goal)
            elif i in end_state:
                cell = GridCell(q, round, CellType.end)
            else:
                cell = GridCell(q, round, CellType.common)
            self.array[row:row+1, col:col+1] = cell.space


def drawQ(Q: np.ndarray, shape: tuple, round :int = 4, start_state: int = None, goal_state: int = None, end_state: list = []):
    grid = Grid(Q, shape, round, start_state, goal_state, end_state)
    for j, rows in enumerate(grid.array):
        print("┌───"*shape[1], end="")
        print("┐")  # 右上角
        print("│", end="")  # 最左边
        for i,col in enumerate(rows):
            print(col, end="")
            print("│", end="")  # 右边
        print()
    print("└───"*shape[1], end="")  # 最下边
    print("┘")


def drawPolicy(Policy: dict, shape: tuple, round:int=4, goal_state: int = None, end_state: list = []):
    if isinstance(Policy, dict):
        array = np.array(list(Policy.values()))
    else:
        array = Policy
    drawQ(array, shape, round=round, goal_state=goal_state, end_state=end_state)


if __name__=="__main__":
    Q = np.array([
        [0.0155,  0.0164,  0.012,  0.015],
        [0.0,   0.0,   0.00,  0.00],
        [5.07,  3.06,  7.86, 2.07],
        [8.73,  8.73,  8.73, 8.73],
        [5.07,  3.06,  3.86, 2.07],
        [3.07,  8.06,  3.86, 2.07],
        [3.07,  3.07,  1.86, 2.07],
        [3.07,  0.06,  3.07, 3.07],
        [0.25,  0.25,  0.25, 0.25]
    ])
    drawQ(Q, (3,3))
