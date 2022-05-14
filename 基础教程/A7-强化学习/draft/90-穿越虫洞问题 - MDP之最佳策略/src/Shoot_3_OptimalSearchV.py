import numpy as np
import Shoot_2_DataModel as dataModel
import math
import Algorithm.Algo_PolicyValueFunction as algo
import common.PrintHelper as helper

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

'''
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
'''

# 把二进制形式的的策略变成onehot编码形式的policy，如 [[0,1],[1,0],[1,0]...]
def create_onehot_policy(binary_actions):
    policy = {}
    for s in range(len(binary_actions)):  # onehot
        policy[s] = [1,0] if binary_actions[s]==0 else [0,1]
    return policy

# 搜索 best_v0 最优策略组合
def find_best_v0_policy(V_values, verbose=True):
    v = np.array(V_values)                    # 列表变成数组
    v0_best = np.max(v[:,0])                  # 获得所有策略组合中 v(s0) 的最大值
    if (verbose):
        helper.print_seperator_line(helper.SeperatorLines.long)
        print("v(s0)的最大 V 函数值 :", v0_best)
    best_ids = np.argwhere(v[:,0] == v0_best) # 获得所有的最大值的策略组合序号
    return best_ids.ravel()


# 创建策略
def create_policy(verbose=True):
    all_policy_in_binary = create_binary_policy()   # 二进制形式
    if (verbose):
        helper.print_seperator_line(helper.SeperatorLines.long)
        print("二进制形式的策略组 : ")
        helper.print_seperator_line(helper.SeperatorLines.middle)
        print(all_policy_in_binary)
    return all_policy_in_binary

# 遍历所有策略组合,计算V,Q
def caculate_all_V_Q(all_policy_in_binary, verbose=True):
    gamma = 1
    max_iteration = 1000
    V_all_policy = []
    Q_all_policy = []
    if (verbose):
        helper.print_seperator_line(helper.SeperatorLines.long)
        print("OneHot形式的策略组与 V 函数值 : ")
        helper.print_seperator_line(helper.SeperatorLines.middle)
    for id, binary_actions in enumerate(all_policy_in_binary):
        policy = create_onehot_policy(binary_actions)      # onehot形式
        env = dataModel.Env(policy)     # 创建环境，代入策略组
        V, Q = algo.calculate_VQ_pi(env, gamma, max_iteration)    # 迭代法计算V,Q
        V_all_policy.append(V)              # 保存每个策略组合的 V 函数结果,便于比较
        Q_all_policy.append(Q)
        if (verbose):
            print(str.format("策略组-{0}:\t{1}", id, policy))
            print(str.format("V 函数值:\t{0}",V))
            helper.print_seperator_line(helper.SeperatorLines.short)
    return V_all_policy, Q_all_policy

# 输出所有满足 best v0 的二进制策略和 V,Q值
def all_best_v0(all_policy_in_binary, V_all_policy, best_ids, verbose=True):
    if (verbose):
        helper.print_seperator_line(helper.SeperatorLines.long)
        print("v(s0)等于最大值的二进制形式的策略组与 V 函数值 :")
        helper.print_seperator_line(helper.SeperatorLines.middle)
    for id in best_ids:
        if (verbose):
            print(str.format("策略组-{0}:\t{1}", id, all_policy_in_binary[id]))
            print(str.format("V 函数值:\t{0}", V_all_policy[id]))
            helper.print_seperator_line(helper.SeperatorLines.short)


if __name__=="__main__":
    # 创建策略
    all_policy_in_binary = create_policy()

    # 遍历所有策略组合
    V_all_policy, Q_all_policy = caculate_all_V_Q(all_policy_in_binary)
    
    # 输出最优策略的价值函数
    best_ids = find_best_v0_policy(V_all_policy)

    # 输出 V
    all_best_v0(all_policy_in_binary, V_all_policy, best_ids)
