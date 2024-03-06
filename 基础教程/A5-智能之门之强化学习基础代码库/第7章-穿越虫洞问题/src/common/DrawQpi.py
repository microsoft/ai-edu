
import numpy as np

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

LEFT_ARROW = 0x25c4 
UP_ARROW = 0x25b2
RIGHT_ARROW = 0x25ba
DOWN_ARROW = 0x25bc
EMPTY_SPACE = 0x0020
CENTER_CROSS = 0x253c
SEP_LINE = 0x2500

dict = {
    "[0]":   "←− ",
    "[1]":   " ↓ ",
    "[2]":   " −→",
    "[3]":   " ↑ ",
    "[0, 1]": "─┐ ",
    "[1, 2]": " ┌─",
    "[2, 3]": " └─",
    "[0, 3]": "─┘ ",
    "[0, 2]": "←−→",
    "[1, 3]": " │ ",
    "[0, 1, 2]": "─┬─",
    "[1, 2, 3]": " ├─",
    "[0, 2, 3]": "─┴─",
    "[0, 1, 3]": "─┤",
    "[0, 1, 2, 3]": "─┼─"
}

class GridCell(object):
    def __init__(self, q, round: int):
        self.space = "  X  "
        self.q = np.round(q, round)
        if np.sum(q) != 0:  # 0 is end state
            best_actions = np.argwhere(self.q == np.max(self.q))
            pos = best_actions.flatten().tolist()
            pos_s = str(pos)
            self.space = dict[pos_s]

class Grid(object):
    def __init__(self, Q, shape: tuple, round: int):
        self.array = np.empty(shape, dtype="<U3")
        for i in range(len(Q)):
            row = (int)(i / shape[1])
            col = (int)(i % shape[1])
            q = Q[i]
            cell = GridCell(q, round)
            self.array[row:row+1, col:col+1] = cell.space


def drawQ(Q: np.ndarray, shape: tuple, round :int = 4):
    grid = Grid(Q, shape, round)
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


def drawPolicy(Policy: dict, shape: tuple, round:int=4):
    array = np.array(list(Policy.values()))
    drawQ(array, shape, round)


if __name__=="__main__":
    Q = np.array([
        [0.0155,  0.0164,  0.012,  0.015],
        [0.0,  0.0,  0.00,  0.00],
        [5.07,  3.06,  7.86 , 2.07],
        [8.73,  8.73,  8.73 , 8.73],
        [5.07,  3.06,  3.86 , 2.07],
        [3.07,  8.06,  3.86 , 2.07],
        [3.07,  3.07,  1.86 , 2.07],
        [3.07,  0.06,  3.07 , 3.07]
    ])
    drawQ(Q, (4,2))
