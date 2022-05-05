
import numpy as np

class GridCell(object):
    def __init__(self, policy):
        self.space = np.zeros((3,5), dtype=int)
        self.space.fill(0x0020)     # 空格
        self.space[1,2] = 0x253c    # 圆圈 O
        self.policy = np.round(policy, 2)

    # policy 是一个1x4的数组或矩阵,如[0.1,0.1,0.4,0.4]
    # 这种情况要在向上和向右的地方画两个箭头
    #  01234
    # 0  ^
    # 1  o >
    # 2
    
        best_actions = np.argwhere(self.policy == np.max(self.policy))
        pos = best_actions.flatten().tolist()
        for a in pos:
            if a == 0:      # left
                self.space[1,0] = 0x25c4 #0x2190
                self.space[1,1] = 0x2015
            elif a == 1:    # up
                self.space[0,2] = 0x25b2 #0x2191
            elif a == 2:    # right
                self.space[1,3] = 0x2015
                self.space[1,4] = 0x25ba #0x2192
            elif a == 3:    # down
                self.space[2,2] = 0x25bc #0x2193
        

class Grid(object):
    def __init__(self, q, shape):
        self.array = np.zeros((shape[0]*3, shape[1]*5), dtype=int)
        for i in range(len(q)):
            row = (int)(i / shape[0])
            col = (int)(i % shape[0])
            policy = q[i]
            cell = GridCell(policy)
            self.array[row*3:row*3+3, col*5:col*5+5] = cell.space

def draw(q, shape):
    grid = Grid(q, shape)
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
    q = np.array([
        [0.49,  0.49,  5.07,  0.58],
        [5.63,  5.63,  5.63,  5.63],
        [5.07,  3.06,  7.86 , 2.07],
        [8.73,  8.73,  8.73 , 8.73]
    ])
    grid = Grid(q,(2,2))
    for rows in grid.array:
        for col in rows:
            print(chr(col), end="")
        print()
