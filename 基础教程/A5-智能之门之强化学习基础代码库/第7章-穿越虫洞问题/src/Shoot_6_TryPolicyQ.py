import numpy as np
import copy
import Shoot_4_DataModel as dataModel
import common.Algo_PolicyEvaluation as algo
import common.CommonHelper as helper

if __name__=="__main__":
    Policy = {      # 原始状态
        0:[0.4,0.6],    # 状态0：[动作 0 概率，动作 1 概率]
        1:[0.4,0.6],    # 状态1：[动作 0 概率，动作 1 概率]
        2:[0.4,0.6],    # 状态2：[动作 0 概率，动作 1 概率]
        3:[0.4,0.6],    # 状态3：[动作 0 概率，动作 1 概率]
        4:[0.4,0.6],    # 状态4：[动作 0 概率，动作 1 概率]
        5:[0.4,0.6]     # 状态5：[动作 0 概率，动作 1 概率]
    }
    gamma = 1
    max_iteration = 1000
    env = dataModel.Env(Policy)
    V,Q = algo.calculate_VQ_pi(env, gamma, max_iteration)
    print("在原始策略下的状态价值函数值 Q:")
    print(np.round(Q[0],3))
    # print("在原始策略下的动作价值函数值 Q:")
    # print("  红     蓝")
    # print(np.round(Q,3))

    # 新策略
    test_policy = np.array([
        [0.3,0.7],  # 修改状态 5 的策略 1
        [0.2,0.8],  # 修改状态 5 的策略 2
        [0.0,1.0],  # 修改状态 5 的策略 3
    ])
    # 每次只修改一个策略,保持其它策略不变,以便观察其影响
    for i in range(3):
        helper.print_seperator_line(helper.SeperatorLines.middle)
        print(str.format("修改状态 0 的策略:{0}", test_policy[i]))
        new_policy = copy.deepcopy(Policy)  # 继承原始策略
        new_policy[5] = test_policy[i]      # 只修改其中一个状态的策略
        env = dataModel.Env(new_policy)
        V,Q = algo.calculate_VQ_pi(env, gamma, max_iteration)
        print("动作价值函数：",np.round(Q[0],3))
