from turtle import st
import numpy as np
import tqdm


# 状态
S_C1 = 0
S_C2 = 1
S_C3 = 2
S_Pass = 3
S_Pub = 4
S_Play = 5
S_Sleep = 6

States = [0,1,2,3,4,5,6]

# 收益向量
# [C1, C2, C3, Pass, Pub, Play, Sleep]
Rewards = [-2, -2, -2, 10, 1, -1, 0]

# 状态转移矩阵
Matrix = np.array(
    [
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2],
        [0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]
)



def episode(start_state, gamma):
    assert (start_state >= S_C1)
    if (start_state == S_Sleep):
        return 0
    curr_state = start_state
    v = Rewards[curr_state]
    while (True):
        next_state = next(curr_state)
        if (next_state == S_Sleep):
            # print("Sleep")
            break
        v += gamma * Rewards[next_state]
        curr_state = next_state
    return v
    
def next(curr_state):
    next_state = np.random.choice(7, p=Matrix[curr_state])
    return next_state

if __name__=="__main__":
    gamma = 0.9
    episodes = 1000
    values = [0] * 7
    for start_state in States:
        q = 0
        for i in tqdm.trange(episodes):
            v = episode(start_state, gamma)
            q += (v - q)/(i+1)
        values[start_state] = q
    
    print(values)
