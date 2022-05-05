import numpy as np
import Shoot_DataModel as dataModel
import math
import Algo_ValueFunctionUnderPolicy as algo

A = [[0,1],[1,0]]


# 给指定的policy的前n/2个填0,后n/2个填1
def fill_number(policy, pos, n):
    for i in range(n):
        if i < n/2:
            policy[i,pos] = 0
        else:
            policy[i,pos] = 1


def create_policy(env):
    n = (int)(math.pow(2, 6))
    policy = np.zeros((n, 6), dtype=np.int32)
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
    all_policy = create_policy(env)
    gamma = 1
    max_iteration = 1000
    V_values = []
    for id, actions in enumerate(all_policy):
        policy = {}
        for s in range(6):
            policy[s] = [0,1] if actions[s]==0 else [1,0]
        print("策略", id)
        print(policy)
        env.Policy = policy
        V, Q = algo.V_in_place_update(env, gamma, max_iteration)
        V_values.append(V)
        print(np.round(V,4))
        print("------")
        

    v = np.array(V_values)
    best_v = np.argwhere(v == np.max(v))
    print(best_v)
    print(np.max(v))
    print(v[best_v[:,0]])