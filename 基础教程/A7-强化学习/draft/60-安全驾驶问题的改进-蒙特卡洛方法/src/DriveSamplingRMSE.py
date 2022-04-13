
import tqdm
import numpy as np
import DriveDataModel as data
import matplotlib.pyplot as plt


# 反向计算G值，记录每个状态的G值，每次访问型
def MC_Sampling_Reverse_Checkpoint(dataModel, start_state, episodes, gamma, checkpoint):
    V = []
    V_value_count_pair = np.zeros((dataModel.num_states, 2))  # state[total value, count of g]
    for episode in tqdm.trange(episodes):
        trajectory = []     # 按顺序 t 保留一幕内的采样序列
        trajectory.append((start_state.value, dataModel.get_reward(start_state)))
        curr_s = start_state
        is_end = False
        while (is_end is False):
            # 从环境获得下一个状态和奖励
            next_s, r, is_end = dataModel.step(curr_s)
            trajectory.append((next_s.value, r))
            curr_s = next_s
        #endwhile
        G = 0
        # 从后向前遍历
        for t in range(len(trajectory)-1, -1, -1):
            s, r = trajectory[t]
            G = gamma * G + r
            V_value_count_pair[s, 0] += G     # 累积总和
            V_value_count_pair[s, 1] += 1     # 累计次数
        #endfor
        if (episode+1)%checkpoint == 0:
            V.append(V_value_count_pair[:,0] / V_value_count_pair[:,1])   # 计算平均值
    #endfor
    V.append(V_value_count_pair[:,0] / V_value_count_pair[:,1])   # 计算平均值
    return V

# MC1的改进，反向计算G值，记录每个状态的G值，每次访问型
def MC_Sampling_Reverse_FirstVisit(dataModel, start_state, episodes, gamma):
    V_value_count_pair = np.zeros((dataModel.num_states, 2))  # state[total value, count of g]
    V_value_count_pair[:,1] = 1 # 避免被除数为0
    for episode in tqdm.trange(episodes):
        trajectory = []     # 一幕内的采样序列
        curr_s = start_state
        trajectory.append((curr_s.value, dataModel.get_reward(start_state)))
        is_end = False
        while (is_end is False):
            # 从环境获得下一个状态和奖励
            next_s, r, is_end = dataModel.step(curr_s)
            #endif
            trajectory.append((next_s.value, r))
            curr_s = next_s
        #endwhile
        num_step = len(trajectory)
        G = 0
        first_visit = set()
        # 从后向前遍历
        for t in range(num_step-1, -1, -1):
            s, r = trajectory[t]
            G = gamma * G + r
            if (s in first_visit):
                continue
            V_value_count_pair[s, 0] += G     # total value
            V_value_count_pair[s, 1] += 1     # count
            first_visit.add(s)
        #endfor
    #endfor
    V = V_value_count_pair[:,0] / V_value_count_pair[:,1]
    return V

def RMSE(a,b):
    err = np.sqrt(np.sum(np.square(a - b))/a.shape[0])
    return err

def test_once():
    episodes = 40000        # 计算 50000 次的试验的均值作为数学期望值
    gamma = 1    # 指定多个折扣因子做试验
    dataModel = data.DataModel()
    checkpoint = 100
    V = MC_Sampling_Reverse_Checkpoint(dataModel, dataModel.S.Start, episodes, gamma, checkpoint)
    print("gamma =", gamma)
    for s in dataModel.S:
        print(str.format("{0}:\t{1:.2f}", s.name, V[-1][s.value]))

    errors = []
    for i in range(len(V)):
        err = RMSE(V[i], dataModel.V_ground_truth)
        errors.append(err)

    return errors

if __name__=="__main__":
    ERRORS = []
    for i in range(10):
        errors = test_once()
        ERRORS.append(errors)

    avg_E = np.mean(ERRORS, axis=0)
    plt.title(str.format("min RMSE={0}", np.min(avg_E)))
    plt.plot(avg_E)    
    plt.grid()
    plt.show()
