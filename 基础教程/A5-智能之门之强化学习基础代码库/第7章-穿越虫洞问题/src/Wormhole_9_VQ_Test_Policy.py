import numpy as np
import Wormhole_2_Data as data              # 数据定义
import common.GridWorld_Model as model           # 模型逻辑
import common.Algo_DP_PolicyEvaluation as algo     # 算法实现
import common.DrawQpi as drawQ                     # 结果输出
import common.CommonHelper as helper

if __name__=="__main__":
    
    test_policies = [
        [1, 0, 0, 0],  # 向左
        [0, 1, 0, 0],  # 向下
        [0, 0, 1, 0]   # 向右
    ]

    for test_policy in test_policies:
        helper.print_seperator_line(helper.SeperatorLines.long)
        env = model.GridWorld(  # 把数据灌入模型中
            # 关于状态的参数
            data.GridWidth, data.GridHeight, data.StartStates, data.EndStates,  
            # 关于动作的参数
            data.Actions, data.Policy, data.Transition,                     
            # 关于奖励的参数
            data.StepReward, data.SpecialReward,                     
            # 关于移动的限制 
            data.SpecialMove, data.Blocks)                        
        # 只修改 s_0 的策略
        env.Policy[0] = test_policy
        gamma = 0.9         # 折扣，在本例中用1.0可以收敛
        V_pi, Q_pi = algo.calculate_VQ_pi(env, gamma)   # 把模型灌入算法中
        V = np.reshape(np.round(V_pi,1), (data.GridWidth, data.GridHeight))
        print("状态价值函数 V:")
        print(V)
        print("S[0]策略:",test_policy)
        print("动作价值函数 Q(0,a):")
        Q = np.round(Q_pi, 1)
        print(Q[0])
        drawQ.drawQ(Q, (data.GridWidth, data.GridHeight), round=2)
