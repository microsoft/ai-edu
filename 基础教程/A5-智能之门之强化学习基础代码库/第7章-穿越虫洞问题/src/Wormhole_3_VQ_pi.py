import numpy as np
import Wormhole_2_Data as data              # 数据定义
import common.GridWorld_Model as model           # 模型逻辑
import common.Algo_PolicyEvaluation as algo     # 算法实现
import common.DrawQpi as drawQ                     # 结果输出
import common.CommonHelper as helper

if __name__=="__main__":
    env = model.GridWorld(  # 把数据灌入模型中
        # 关于状态的参数
        data.GridWidth, data.GridHeight, data.StartStates, data.EndStates,  
        # 关于动作的参数
        data.Actions, data.Policy, data.Transition,                     
        # 关于奖励的参数
        data.StepReward, data.SpecialReward,                     
        # 关于移动的限制 
        data.SpecialMove, data.Blocks)                        

    gamma = 0.9         # 折扣，在本例中用1.0可以收敛
    V_pi, Q_pi = algo.calculate_VQ_pi(env, gamma)   # 把模型灌入算法中

    helper.print_seperator_line(helper.SeperatorLines.long)

    print("随机策略下的状态价值函数 V:")
    V = np.reshape(np.round(V_pi,1), (data.GridWidth, data.GridHeight))
    print(V)

    helper.print_seperator_line(helper.SeperatorLines.long)

    print("随机策略下的动作价值函数 Q:")
    Q = np.reshape(np.round(Q_pi, 1), (data.GridWidth, data.GridHeight, len(data.Actions)))
    print(Q)

    # 字符图形化显示
    helper.print_seperator_line(helper.SeperatorLines.long)

    drawQ.drawQ(Q_pi, (data.GridWidth, data.GridHeight), round=1)
