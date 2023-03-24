import multiprocessing as mp
import tqdm
import numpy as np
import DriveDataModel as data
import matplotlib.pyplot as plt

# 多状态同时更新的蒙特卡洛采样
# 注意输入V有初始状态
# constant-alpha
def MC_Incremental_Update(dataModel, start_state, episodes, alpha, gamma):
    V = np.zeros((dataModel.num_states))
    for episode in tqdm.trange(episodes):
        trajectory = []
        if (start_state is None):
            curr_s = dataModel.random_select_state()
        else:
            curr_s = start_state

        trajectory.append((curr_s.value, dataModel.get_reward(curr_s)))
        is_end = False
        while (is_end is False):
            # 从环境获得下一个状态和奖励
            next_s, R, is_end = dataModel.step(curr_s)
            #endif
            trajectory.append((next_s.value, R))
            curr_s = next_s

        # calculate G_t
        num_step = len(trajectory) 
        G = 0
        # 从后向前遍历
        for t in range(num_step-1, -1, -1):
            S, R = trajectory[t]
            G = gamma * G + R
            V[S] = V[S] + alpha * (G - V[S])
        #endfor
    #endfor
    return V

def MultiProcess(alphas):
    pool = mp.Pool(processes=4) # 指定合适的进程数量
    ERRORS = []
    results = []
    for alpha in alphas:    # 遍历状态集中的每个状态作为起始状态
        results.append(pool.apply_async(test_once, args=(alpha,)))
    pool.close()
    pool.join()
    for i in range(len(alphas)):
        avg_e = results[i].get()
        ERRORS.append(avg_e)
    
    return ERRORS   # 4 个 alpha 对应的平均 error

# 每隔100幕保存一个中间结果
def MC_Incremental_Update_Checkpoint(dataModel, start_state, episodes, alpha, gamma, checkpoint):
    Vs = []
    V = np.zeros((dataModel.num_states))
    for episode in tqdm.trange(episodes):
        trajectory = []
        if (start_state is None):
            curr_s = dataModel.random_select_state()
        else:
            curr_s = start_state

        trajectory.append((curr_s.value, dataModel.get_reward(curr_s)))
        is_end = False
        while (is_end is False):
            # 从环境获得下一个状态和奖励
            next_s, R, is_end = dataModel.step(curr_s)
            #endif
            trajectory.append((next_s.value, R))
            curr_s = next_s

        # calculate G_t
        num_step = len(trajectory) 
        G = 0
        # 从后向前遍历
        for t in range(num_step-1, -1, -1):
            S, R = trajectory[t]
            G = gamma * G + R
            V[S] = V[S] + alpha * (G - V[S])
        #endfor
        if (episode+1)%checkpoint == 0:
            Vs.append(V.copy())
    #endfor
    return Vs

def RMSE(a,b):
    err = np.sqrt(np.sum(np.square(a - b))/a.shape[0])
    return err

# 针对一个alpha做10次的平均error
def test_once(alpha):
    ERRORS = []
    for i in range(2):
        episodes = 20000        # 计算 10000 次的试验的均值作为数学期望值
        checkpoint = 100
        gamma = 1    # 指定多个折扣因子做试验
        dataModel = data.DataModel()
        errors = []
        Vs = MC_Incremental_Update_Checkpoint(dataModel, dataModel.S.Start, episodes, alpha, gamma, checkpoint)
        for result in Vs:
            err = RMSE(result, dataModel.V_ground_truth)
            errors.append(err)
        ERRORS.append(errors)
    avg_E = np.mean(ERRORS, axis=0)
    return avg_E

if __name__=="__main__":
    alphas = [0.0005,0.001,0.002,0.005,0.01]
    avgErros = MultiProcess(alphas)
    for i, avg_E in enumerate(avgErros):
        plt.plot(avg_E, label=str.format("alpha={0}, min RMSE={1:.3f}", alphas[i], np.min(avg_E)))
    plt.legend()
    plt.grid()
    plt.show()
