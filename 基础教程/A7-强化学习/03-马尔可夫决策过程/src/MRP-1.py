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

def run(gamma):
    V_curr = [0.0] * 7
    V_next = [0.0] * 7
    count = 0
    while (True):
        # 遍历每一个 state 作为 start_state
        for start_state in States:
            # 得到转移概率
            prob = Matrix[start_state.value]
            v_sum = 0
            # 计算下一个状态的 转移概率*状态值 的 和 v
            for next_state_value in range(len(prob)):
                # if (prob[next_state] > 0.0):
                v_sum += prob[next_state_value] * V_next[next_state_value]
            # end for
            V_curr[start_state.value] = Rewards[start_state.value] + gamma * v_sum
        # end for
        # 检查收敛性
        if np.allclose(V_next, V_curr):
            break
        # 把 V_curr 赋值给 V_next
        V_next = V_curr.copy()
        count += 1
    # end while
    print(count)
    return V_next

if __name__=="__main__":
    gamma = 0.9
    v = run(gamma)
    for start_state in States:
        print(start_state, v[start_state.value])
