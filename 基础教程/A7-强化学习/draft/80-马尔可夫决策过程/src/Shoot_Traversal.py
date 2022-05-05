import numpy as np
from enum import Enum
import copy
import Shoot_DataModel as dataModel
import math


A = [[0,1],[1,0]]


# 给指定的policy的前n/2个填0,后n/2个填1
def fill_number(policy, pos, n):
    for i in range(n):
        if i < n/2:
            policy[i,pos] = 0
        else:
            policy[i,pos] = 1


def create_policy(env):
    n = (int)(math.pow(2, env.nS))
    policy = np.zeros((n, env.nS), dtype=np.int32)
    pos = 0
    while True:
        count = (int)(math.pow(2,pos))
        for i in range(count):
            start = i * n
            end = (i+1) * n
            fill_number(policy[start:end], pos, n)
        n = (int)(n/2)
        pos += 1
        if n == 1:
            break

    return policy

if __name__=="__main__":
    
    env = dataModel.Env()
    actions = create_policy(env)
    
    for a in actions:
        policy = {}
        for s in range(env.nS):
            policy[s] = [0,1] if a==0 else [1,0]
            