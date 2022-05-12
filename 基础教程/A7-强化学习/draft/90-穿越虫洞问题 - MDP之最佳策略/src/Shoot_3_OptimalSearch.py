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
        for item in list0:
            list2.append([item])
        for item in list1:
            list2.append([item])
        count += 1
    print(list2)

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
    best_ids = np.argwhere(v[:,0] == v0_best) # 获得所有的最大值的策略组合序号
    return best_ids

if __name__=="__main__":
    all_policy_in_binary = create_binary_policy()   # 二进制形式
    print("二进制形式的策略组 : ")
    print("-"*30)
    print(all_policy_in_binary)
    print("="*30)
    # 遍历所有策略组合
    gamma = 1
    max_iteration = 1000
    V_values = []
    print("OneHot形式的策略组与状态价值函数 : ")
    print("-"*20)
    for id, actions in enumerate(all_policy_in_binary):
        policy = create_onehot_policy(actions)      # onehot形式
        print(str.format("策略组-{0}:\t{1}", id, policy))
        env = dataModel.Env(policy)     # 创建环境，代入策略组合
        V, Q = algo.calculate_Vpi_Qpi(env, gamma, max_iteration)    # 迭代法计算V,Q
        V_values.append(V)              # 保存每个策略组合的价值函数结果,便于比较
        print(str.format("状态价值函数:\t{0}",V))
        print("-"*10)
    
    print("="*40)
    # 输出最优策略的价值函数
    best_ids = find_best_policy(V_values, all_policy_in_binary)
    print("二进制形式的最优策略组与最优价值函数 :")
    print("-"*20)
    for id in best_ids:
        policy = create_onehot_policy(all_policy_in_binary[id[0]])      # onehot形式
        print(str.format("最优策略组-{0}:\t{1}", id[0], all_policy_in_binary[id[0]]))
        env = dataModel.Env(policy)     # 创建环境，代入策略组合
        V, Q = algo.calculate_Vpi_Qpi(env, gamma, max_iteration)    # 迭代法计算V,Q
        print(str.format("最优状态价值函数:\t{0}", V))
        print("最优动作价值函数:")
        print(Q)
        print("-"*10)
