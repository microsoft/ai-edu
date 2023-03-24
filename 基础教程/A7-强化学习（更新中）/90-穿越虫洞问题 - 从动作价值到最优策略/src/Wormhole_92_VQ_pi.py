import numpy as np
import Wormhole_91_Data as data              # 数据定义
import common.GridWorld_Model as model           # 模型逻辑
import Algorithm.Algo_PolicyValueFunction as algo     # 算法实现
import common.DrawQpi as drawQ                     # 结果输出
import common.CommonHelper as helper

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

    gamma = 0.9         # 折扣，在本例中用1.0可以收敛
    iteration = 1000    # 算法最大迭代次数
    V_pi, Q_pi = algo.calculate_VQ_pi(env, gamma, iteration)  # 原地更新的迭代算法
    helper.print_seperator_line(helper.SeperatorLines.long)
    print("V_pi")
    V = np.reshape(np.round(V_pi,1), (data.GridWidth, data.GridHeight))
    print(V)
    helper.print_seperator_line(helper.SeperatorLines.long)
    print("Q_pi")
    print(np.round(Q_pi,1))
    # 字符图形化显示
    helper.print_seperator_line(helper.SeperatorLines.long)
    drawQ.draw(Q_pi, (data.GridWidth, data.GridHeight))
