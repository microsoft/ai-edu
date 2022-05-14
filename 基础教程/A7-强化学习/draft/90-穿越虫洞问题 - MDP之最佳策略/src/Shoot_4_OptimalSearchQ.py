import common.PrintHelper as helper
from Shoot_3_OptimalSearchV import *

# 输出所有满足 best v0 的二进制策略和 V,Q值
def all_best_v0(all_policy_in_binary, V_all_policy, Q_all_policy, best_ids, verbose=True):
    helper.print_seperator_line(helper.SeperatorLines.long)
    print("v(s0)等于最大值的二进制形式的策略组与 V 函数值 :")
    helper.print_seperator_line(helper.SeperatorLines.middle)
    for id in best_ids:
        print(str.format("策略组-{0}:\t{1}", id, all_policy_in_binary[id]))
        print(str.format("V 函数值:\t{0}", V_all_policy[id]))
        print("Q 函数值:")
        print(Q_all_policy[id])
        helper.print_seperator_line(helper.SeperatorLines.short)


if __name__=="__main__":
    # 创建策略
    all_policy_in_binary = create_policy(verbose=False)

    # 遍历所有策略组合
    V_all_policy, Q_all_policy = caculate_all_V_Q(all_policy_in_binary, verbose=False)
    
    # 输出最优策略的价值函数
    best_ids = find_best_v0_policy(V_all_policy, verbose=False)

    # 输出 Q
    all_best_v0(all_policy_in_binary, V_all_policy, Q_all_policy, best_ids)
