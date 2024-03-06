
# 策略迭代，包括策略评估和策略改进
# 策略评估使用 Algo_PolicyEvaluation.py 中的 calculate_VQ_pi 计算当前策略的 V,Q

import Wormhole_2_Data as data
import common.GridWorld_Model as model           
import common.DrawQpi as drawQ      
import common.Algo_DP_PolicyIteration as PolicyIteration
import numpy as np

if __name__=="__main__":
    env = model.GridWorld(
        # 关于状态的参数
        data.GridWidth, data.GridHeight, data.StartStates, data.EndStates,  
        # 关于动作的参数
        data.Actions, data.Policy, data.Transition,                     
        # 关于奖励的参数
        data.StepReward, data.SpecialReward,                     
        # 关于移动的限制 
        data.SpecialMove, data.Blocks)                        

    gamma = 0.9
    print("初始策略")
    print(env.Policy)
    final_policy, V, Q = PolicyIteration.policy_iteration(env, gamma, verbose=True)
    print("最终策略")
    print(final_policy)
    drawQ.drawPolicy(env.Policy, (data.GridWidth, data.GridHeight), round=1)

    print("最终策略下的状态价值函数 V:")
    V = np.reshape(np.round(V,1), (data.GridWidth, data.GridHeight))
    print(V)
