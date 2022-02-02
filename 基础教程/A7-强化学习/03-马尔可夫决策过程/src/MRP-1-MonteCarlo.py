import math
from turtle import st
import numpy as np
from torch import ne
import tqdm
from enum import Enum
import multiprocessing as mp

# 状态
class States(Enum):
    Class1 = 0
    Class2 = 1
    Class3 = 2
    Pass = 3
    Pub = 4
    Play = 5
    Sleep = 6

# 收益向量
# [Sleep, Pass, C3, C2, C1, Pub, Play]
#Rewards = [0, 10, -2, -2, -2, 1, -1]
Rewards = [-2, -2, -2, 10, 1, -1, 0]
'''
# 状态转移矩阵
Matrix = np.array(
    [
        #Slp  Pas  Cl3  Cl2  Cl1  Pub  Ply
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    #Slp
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    #Pas
        [0.0, 0.6, 0.0, 0.0, 0.0, 0.4, 0.0],    #Cl3
        [0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0],    #Cl2
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5],    #Cl1
        [0.0, 0.0, 0.4, 0.4, 0.2, 0.0, 0.0],    #Pub
        [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.9]     #Ply
    ]
)

'''
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


def run(start_state, episodes, gamma):
    sum_g = 0
    for episode in tqdm.trange(episodes):
        if (start_state == States.Sleep):
            return 0
        curr_state = start_state
        g = get_reward(curr_state)
        power = 1
        while (True):
            next_state = get_next_state(curr_state)
            r = get_reward(next_state)
            g += math.pow(gamma, power) * r
            if (next_state == States.Sleep):
                break
            else:
                power += 1
                curr_state = next_state
        # end while
        sum_g += g
    # end for
    v = sum_g / episodes
    return v
        
    
def get_next_state(curr_state):
    next_state_value = np.random.choice(7, p=Matrix[curr_state.value])
    return States(next_state_value)

def get_reward(curr_state):
    return Rewards[curr_state.value]

if __name__=="__main__":
    gamma = 0.9
    episodes = 10000

    pool = mp.Pool(processes=6)
    Vs = []
    results = []
    for start_state in States:
        results.append(pool.apply_async(run, args=(start_state,episodes,gamma,)))
    pool.close()
    pool.join()
    for i in range(len(results)):
        v = results[i].get()
        Vs.append(v)

    for state in States:
        print("{}:{}".format(state, Vs[state.value]))
