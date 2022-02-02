import math
from turtle import st
import numpy as np
from torch import ne
import tqdm
from enum import Enum
import multiprocessing as mp

# 状态定义
class States(Enum):
    Class1 = 0
    Class2 = 1
    Class3 = 2
    Pass = 3
    Pub = 4
    Play = 5
    Sleep = 6

# 收益向量
# [Class1, Class2, Class3, Pass, Pub, Play, Sleep]
Rewards = [-2, -2, -2, 10, 1, -1, 0]

Matrix = np.array(
    [   #Cl1  Cl2  Cl3  Pas  Pub  Ply  Slp
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0], # Class1
        [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2], # CLass2
        [0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0], # Class3
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], # Pass
        [0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0], # Pub
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0], # Play
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Sleep
    ]
)

def InvMatrix(gamma):
    num_state = Matrix.shape[0]
    I = np.eye(num_state)
    tmp1 = I - gamma * Matrix
    tmp2 = np.linalg.inv(tmp1)
    values = np.dot(tmp2, Rewards)
    print(values)

if __name__=="__main__":
    gamma = 0.9
    InvMatrix(gamma)
