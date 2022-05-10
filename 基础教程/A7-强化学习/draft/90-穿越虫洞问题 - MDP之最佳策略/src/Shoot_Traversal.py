import numpy as np
import Shoot_2_DataModel as dataModel
import math
import Algo_PolicyValueFunction as algo
import copy

# 给指定的policy的前n/2个填0,后n/2个填1
def fill_number(policy, pos, n):
    for i in range(n):
        if i < n/2:
            policy[i,pos] = 0
        else:
            policy[i,pos] = 1


def create_policy():
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


def test():
    list2 = []
    n = 3
    count = 0
    while count < n:
        list0 = copy.deepcopy(list2)
        list1 = copy.deepcopy(list2)
        if len(list0) == 0:
            list0.append(0)
            list1.append(1)
        else:
            for item in list0:
                item.append(0)
            for item in list1:
                item.append(1)
        list2 = []
        list2.append(list0)
        list2.extend(list1)
        count += 1
    print(list2)

if __name__=="__main__":
    
    #test()
    #exit(0)

    all_policy = create_policy()
    env = dataModel.Env(all_policy)

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
        V, Q = algo.calculate_Vpi_Qpi(env, gamma, max_iteration)
        V_values.append(V)
        print(np.round(V,4))
        print("------")
        

    v = np.array(V_values)
    best_v = np.argwhere(v == np.max(v))
    print(best_v)
    print(np.max(v))
    print(v[best_v[:,0]])