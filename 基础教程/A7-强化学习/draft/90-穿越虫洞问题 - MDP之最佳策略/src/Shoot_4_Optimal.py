from email import policy
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


def create_binary_policy():
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



# 创建onehot编码形式的policy，如 [[0,1],[1,0],[1,0]...]
def create_onehot_policy(actions):
    policy = {}
    for s in range(len(actions)):  # onehot
        policy[s] = [1,0] if actions[s]==0 else [0,1]
    return policy

# 搜索最优策略组合
def find_best_policy(V_values, all_policy_in_binary):
    v = np.array(V_values)                    # 列表变成数组
    v0_best = np.max(v[:,0])                  # 获得所有策略组合中 v(s0) 的最大值
    print("v(s0)的最优价值函数 :", v0_best)
    print("="*40)
    #print("二进制形式的最优策略组合与价值函数 :")
    #print("-"*20)
    best_ids = np.argwhere(v[:,0] == v0_best) # 获得所有的最大值的策略组合序号
    #for id in best_ids:
        #print(str.format("最优策略组合({0}):\t{1}", id[0], all_policy_in_binary[id[0]]))
        #print(str.format("最优状态价值函数:\t{0}", v[id][0]))
        #print("-"*10)
    return best_ids

if __name__=="__main__":
    all_policy_in_binary = create_binary_policy()   # 二进制形式
    gamma = 1
    max_iteration = 1000
    V_values = []
    print("OneHot形式的策略组合与状态价值函数 : ")
    print("-"*20)
    for id in range(59,64):
        actions = all_policy_in_binary[id]
        policy = create_onehot_policy(actions)      # onehot形式
        print(str.format("策略组合({0}):\t{1}", id, policy))
        env = dataModel.Env(policy)     # 创建环境，代入策略组合
        V, Q = algo.calculate_Vpi_Qpi(env, gamma, max_iteration)    # 迭代法计算V,Q
        V_values.append(V)              # 保存每个策略组合的价值函数结果,便于比较
        print(str.format("状态价值函数:\t{0}",V[0]))
        print(str.format("动作价值函数:\n{0}",Q))
        print("-"*10)
    
