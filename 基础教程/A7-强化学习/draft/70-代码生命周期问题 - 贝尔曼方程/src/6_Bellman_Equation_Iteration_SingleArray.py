import numpy as np
from enum import Enum

# 状态
class States(Enum):
    Bug = 0
    Coding = 1
    Test = 2
    Review = 3
    Refactor = 4
    Merge = 5
    End = 6

# 奖励向量，顺序与States定义一致
#Rewards = [-3, -2, -1, 0, 5, -4, 0]
# 奖励向量 缺陷 编码 测试 审查 重构 合并 结束
Rewards = [-3, 0,   +1, +3,  +2, -1,  0]

# 用字典代替状态转移矩阵
D = {
    States.Bug: [(States.Bug, 0.7),(States.Coding, 0.3)],
    States.Coding: [(States.Bug, 0.6),(States.Test, 0.4)],
    States.Test: [(States.Review, 0.9),(States.End, 0.1)],
    States.Review: [(States.Refactor, 0.2),(States.Merge, 0.8)],
    States.Refactor: [(States.Coding, 0.2),(States.Test, 0.5),(States.Review, 0.3)],
    States.Merge: [(States.End, 1.0)],
    States.End: [(States.End, 1.0)]
}

class DataModel(object):
    def __init__(self):
        self.D = D                          # 状态转移字典
        self.R = Rewards                    # 奖励
        self.S = States                     # 状态集
        self.N = len(self.S)                # 状态数量
        self.E = [self.S.End]      # 终止状态集

    def get_next(self, curr_s):
        list_state_prob = self.D[curr_s]    # 根据当前状态返回可用的下游状态及其概率
        return list_state_prob


# 贝尔曼方程单数组就地更新
def Bellman_iteration_single_array(dataModel, gamma):
    print("---单数组就地更新法---")
    V = np.zeros(dataModel.N)
    count = 0
    while (count < 1000):   # 1000 是随意指定的一个比较大的数，避免不收敛而导致while无限
        count += 1
        V_old = V.copy()
        # 遍历每一个 state 作为 curr_state
        for curr_state in dataModel.S:
            # 得到转移概率
            list_state_prob  = dataModel.get_next(curr_state)
            # 计算 \sum(P·V)
            v_sum = 0
            for next_state, next_prob in list_state_prob:
                v_sum += next_prob * V[next_state.value]
            # 计算 V = R + gamma * \sum(P·V)
            V[curr_state.value] = dataModel.R[curr_state.value] + gamma * v_sum
        # 检查收敛性
        if np.allclose(V, V_old):
            break
    print("迭代次数 :", count)
    return V

def print_V(V):
    print(V)
    vv = np.around(V,3)
    for s in dataModel.S:
        print(str.format("{0}:\t{1}", s.name, vv[s.value]))

if __name__=="__main__":
    dataModel = DataModel()
    gamma = 1

    V = Bellman_iteration_single_array(dataModel, gamma)
    print_V(V)