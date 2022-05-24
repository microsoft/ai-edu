import numpy as np
import Wormhole_0_Data as data              # 数据定义
import GridWorld_Model as model             # 模型逻辑
import Algorithm.Algo_OptimalValueFunction as algo    # 算法实现
import common.DrawQpi as drawQ                     # 结果输出

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
    #model.print_P(env.P_S_R)
    gamma = 0.9 # 折扣，在本例中用1.0可以收敛，但是用0.9比较保险
    iteration = 1000    # 算法最大迭代次数
    V_star, Q_star = algo.calculate_VQ_star(env, gamma, iteration)  # 原地更新的迭代算法
    print("V*")
    V = np.reshape(np.round(V_star,1), (data.GridWidth, data.GridHeight))
    print(V)
    print("Q*")
    print(np.round(Q_star,1))
    # 字符图形化显示
    drawQ.draw(Q_star, (data.GridWidth, data.GridHeight))
