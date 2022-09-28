
import numpy as np

LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

LEFT_ARROW = 0x25c4 
UP_ARROW = 0x25b2
RIGHT_ARROW = 0x25ba
DOWN_ARROW = 0x25bc
EMPTY_SPACE = 0x0020
CENTER_CROSS = 0x253c
SEP_LINE = 0x2500

class GridCell(object):
    def __init__(self, q):
        self.space = np.zeros((3,5), dtype=int)
        self.space.fill(EMPTY_SPACE)     # 空格
        self.space[1,2] = CENTER_CROSS   # 中心
        self.q = np.round(q, 4)

    # policy 是一个1x4的数组或矩阵,如[0.1,0.1,0.4,0.4]
    # 这种情况要在向上和向右的地方画两个箭头
    #  01234
    # 0  ^
    # 1  o >
    # 2
        if np.sum(q) != 0:
            best_actions = np.argwhere(self.q == np.max(self.q))
            pos = best_actions.flatten().tolist()
            for action in pos:
                if action == LEFT:      # left
                    self.space[1,0] = LEFT_ARROW
                    self.space[1,1] = SEP_LINE
                elif action == UP:    # up
                    self.space[0,2] = UP_ARROW
                elif action == RIGHT:    # right
                    self.space[1,3] = SEP_LINE
                    self.space[1,4] = RIGHT_ARROW
                elif action == DOWN:    # down
                    self.space[2,2] = DOWN_ARROW
        

class Grid(object):
    def __init__(self, Q, shape):
        self.array = np.zeros((shape[0]*3, shape[1]*5), dtype=int)
        for i in range(len(Q)):
            row = (int)(i / shape[1])
            col = (int)(i % shape[1])
            q = Q[i]
            cell = GridCell(q)
            self.array[row*3:row*3+3, col*5:col*5+5] = cell.space

def draw(Q, shape):
    grid = Grid(Q, shape)
    for j, rows in enumerate(grid.array):
        if (j % 3 == 0):
            print("+-----"*shape[1], end="")
            print("+")
        print("|", end="")
        for i,col in enumerate(rows):
            print(chr(col), end="")
            if ((i+1) % 5 == 0):
                print("|", end="")
        print()
    print("+-----"*shape[1])

if __name__=="__main__":
    Q = np.array([
        [0.0155,  0.0164,  0.012,  0.015],
        [0.0,  0.0,  0.00,  0.00],
        [5.07,  3.06,  7.86 , 2.07],
        [8.73,  8.73,  8.73 , 8.73]
    ])
    draw(Q, (2,2))
