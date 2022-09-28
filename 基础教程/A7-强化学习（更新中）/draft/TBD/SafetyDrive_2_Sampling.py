
import tqdm
import numpy as np
import SafetyDrive_0_DataModel as data
import time

# 多次采样获得回报 G 的数学期望，即状态价值函数 v(start_state)
def ReverseCalculate(dataModel, start_state, episodes, gamma):
    Vs_value = np.zeros((dataModel.nS))
    Vs_count = np.zeros((dataModel.nS))
    # 循环多幕
    for episode in tqdm.trange(episodes):
        trajectory = []
        s = start_state # 把给定的起始状态作为当前状态
        is_end = False
        while not is_end:
            s_next, reward, is_end = dataModel.step(s)
            trajectory.append((reward, s_next))
            s = s_next
        # end while
        num_step = len(trajectory)
        g = 0
        # 从后向前遍历，因为 G = R_t+1 + gamma * R_t + gamme^2 * R_t-1 + ...
        for t in range(num_step-1, -1, -1):
            reward, s = trajectory[t]
            g = gamma * g + reward
            Vs_value[s.value] += g     # total value
            Vs_count[s.value] += 1     # count        
    # end for
    V = Vs_value / Vs_count
    return V

if __name__=="__main__":
    start = time.time()
    episodes = 10000        # 计算 10000 次的试验的均值作为数学期望值
    gamma = 1               # 折扣因子
    dataModel = data.DataModel()
    V = ReverseCalculate(dataModel, dataModel.S.Start, episodes, gamma)
    print("gamma =", gamma)
    for s in dataModel.S:
        print(str.format("{0}:\t{1}", s.name, V[s.value]))
