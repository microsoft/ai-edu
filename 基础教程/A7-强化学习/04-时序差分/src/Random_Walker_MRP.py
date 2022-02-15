import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import Algorithm_MRP as algoMRP
import Random_Walker_MRP as ds

class Action(Enum):
    MOVE_LEFT = 0         # left
    MOVE_RIGHT = 1       # right


'''
   [ ]-0-(A)-0-(B)-0-(C)-0-(D)-0-(E)-1-[ ]
s = 0     1     2     3     4     5     6
'''

# 状态
class States(Enum):
    Resturant = 0
    RoadA = 1
    RoadB = 2
    RoadC = 3
    RoadD = 4
    RoadE = 5
    Home = 6

Rewards = [0,0,0,0,0,0,1]

Matrix = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]
)


'''
def MRP():
    V = np.zeros(7)
    gamma = 1
    num_iteration = 100

    for i in range(num_iteration):
        for state in range(7):  # 0~6
            value = 0
            for next_state in range(7): # 0~6
                prob = Pss[state, next_state]
                if (prob > 0.0):
                    # math: \sum Pss'V(s')
                    value += prob * V[next_state]
                #endif
            #endfor
            # math: V(s) = R_s + \gamma \sum Pss'V(s')
            V[state] = Rewards[state] + gamma * value
        #endfor
    #endfor
    print("{0}".format(V))

def InvMatrix(gamma):
    num_state = Pss.shape[0]
    I = np.eye(num_state)
    tmp1 = I - gamma * Pss
    tmp2 = np.linalg.inv(tmp1)
    values = np.dot(tmp2, Rewards)
    print("{0}/6".format(values*6))
'''

if __name__=="__main__":
    gamma = 1
    vs = algoMRP.Matrix(ds, gamma)
    print(vs)
