import numpy as np
import Wormhole_0_Data as data              # 数据定义
import GridWorld_0_Model as model           # 模型逻辑
import Algo_PolicyValueFunction as algo     # 算法实现
import DrawQpi as drawQ                     # 结果输出

if __name__=="__main__":
    env = model.GridWorld(
        # 关于状态的参数
        data.GridWidth, data.GridHeight, data.StartStates, data.EndStates,  
        # 关于动作的参数
        data.Actions, data.Policy, data.SlipProbs,                     
        # 关于奖励的参数
        data.StepReward, data.SpecialReward,                     
        # 关于移动的限制 
        data.SpecialMove, data.Blocks)                        

    gamma = 0.9
    iteration = 1000
    V_pi, Q_pi = algo.V_in_place_update(env, gamma, iteration)
    print("V_pi")
    V = np.reshape(np.round(V_pi,2), (data.GridWidth, data.GridHeight))
    print(V)
    print("Q_pi")
    print(np.round(Q_pi,2))

    drawQ.draw(Q_pi, (data.GridWidth, data.GridHeight))
