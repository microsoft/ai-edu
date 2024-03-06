import numpy as np
from enum import Enum
import common_helper as helper

# 状态
class States(Enum):
    Bug = 0
    Coding = 1
    Test = 2
    Review = 3
    Refactor = 4
    Merge = 5
    End = 6

# 奖励向量 缺陷 编码 测试 审查 重构 合并 结束
Rewards = [-3, 0,   +1, +3,  +2, -1,  0]

# 用字典代替状态转移矩阵
D = {
    States.Bug:     [(States.Bug, 0.2),     (States.Coding, 0.8)],
    States.Coding:  [(States.Bug, 0.6),     (States.Test, 0.4)],
    States.Test:    [(States.Coding, 0.1),  (States.Review, 0.9)],
    States.Review:  [(States.Coding, 0.1),  (States.Refactor, 0.2), (States.Merge, 0.7)],
    States.Refactor:[(States.Coding, 0.2),  (States.Test, 0.5),     (States.Review, 0.3)],
    States.Merge:   [(States.End, 1.0)],
    States.End:     [(States.End, 1.0)]
}

class DataModel(object):
    def __init__(self):
        self.D = D                          # 状态转移字典
        self.R = Rewards                    # 奖励
        self.S = States                     # 状态集
        self.N = len(self.S)                # 状态数量
        self.E = [self.S.End]      # 终止状态集

    def get_next(self, s):
        list_state_prob = self.D[s]    # 根据当前状态返回可用的下游状态及其概率
        return list_state_prob


# 贝尔曼方程单数组更新
def Bellman_iteration(dataModel, gamma, max_iteration):
    helper.print_seperator_line(helper.SeperatorLines.long, info="贝尔曼方程单数组更新法")
    V = np.zeros(dataModel.N)
    count = 0
    while (count < max_iteration): 
        count += 1
        V_old = V.copy()
        # 遍历每一个 state 作为 curr_state
        for s in dataModel.S:
            # 得到转移概率
            list_state_prob  = dataModel.get_next(s)
            temp = 0
            for s_next, p_s_snext in list_state_prob:
                temp += p_s_snext * V[s_next.value]  # 计算 \sum(p·v)
            # 计算 V = R + gamma * \sum(p·v)
            V[s.value] = dataModel.R[s.value] + gamma * temp
        # 检查收敛性
        if np.allclose(V, V_old):
            break
    print("迭代次数 :", count)
    return V

if __name__=="__main__":
    dataModel = DataModel()
    gamma = 1
    max_iteration = 1000
    V = Bellman_iteration(dataModel, gamma, max_iteration)
    helper.print_V(dataModel, V)
