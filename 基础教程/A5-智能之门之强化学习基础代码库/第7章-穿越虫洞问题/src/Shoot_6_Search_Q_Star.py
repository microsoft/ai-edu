import common.CommonHelper as helper
import Shoot_4_Search_V_Star as svs
import numpy as np


# 输出所有满足 best v0 的二进制策略和 V,Q值
def all_best_q(all_policy_in_binary, V_all_policy, Q_all_policy, best_ids, verbose=True):
    helper.print_seperator_line(helper.SeperatorLines.long)
    print("q(s,a)等于最大值的二进制形式的策略组与 Q 函数值 :")
    helper.print_seperator_line(helper.SeperatorLines.middle)
    for id in best_ids:
        print(str.format("策略组-{0}:\t{1}", id, all_policy_in_binary[id]))
        print("V 函数值:")
        print(V_all_policy[id])
        print("Q 函数值:")
        print(Q_all_policy[id])
        helper.print_seperator_line(helper.SeperatorLines.short)

# 搜索 best_v0 最优策略组合
def find_best_q0_policy(Q_values, s, a, verbose=True):
    q = np.array(Q_values)                    # 列表变成数组
    q0_best = np.max(q[:,s,a])                  # 获得所有策略组合中 v(s0) 的最大值
    if (verbose):
        helper.print_seperator_line(helper.SeperatorLines.long)
        print("q(s,a)的最大 Q 函数值 :", q0_best)
    best_ids = np.argwhere(q[:,s,a] == q0_best) # 获得所有的最大值的策略组合序号
    return best_ids.ravel(), q0_best


if __name__=="__main__":
    # 创建策略
    all_policy_in_binary = svs.create_policy(verbose=False)

    # 遍历所有策略组合
    V_all_policy, Q_all_policy = svs.calculate_all_V_Q(all_policy_in_binary, verbose=False)
    
    # 输出最优策略的价值函数
    best_ids, max_q = find_best_q0_policy(Q_all_policy, 0, 0, verbose=False)
    print(max_q, best_ids)
    best_ids, max_q = find_best_q0_policy(Q_all_policy, 0, 1, verbose=False)
    print(max_q, best_ids)

    # 输出 Q
    all_best_q(all_policy_in_binary, V_all_policy, Q_all_policy, [19,23,51,55])
