
import tqdm
import multiprocessing as mp
import math
import numpy as np
import StudentData as data


def Sampling_MultiProcess(ds, episodes, gamma):
    pool = mp.Pool(processes=4)
    V = np.zeros((ds.num_states))
    results = []
    for start_state in ds.S:
        results.append(pool.apply_async(Sampling, 
                args=(ds, start_state, episodes, gamma,)
            )
        )
    pool.close()
    pool.join()
    for i in range(len(results)):
        v = results[i].get()
        V[i] = v

    return V


def RMSE(a, b):
    err = np.sqrt(np.sum(np.square(a-b))/a.shape[0])
    return err


def Sampling_with_history(ds, start_state, episodes, gamma):
    G_history = []

    if (start_state in ds.end_states):
        # 最后一个状态也可能有reward值
        return ds.R[start_state.value], None
    
    for episode in tqdm.trange(episodes):
        curr_s = start_state
        G = ds.get_reward(curr_s)
        power = 1
        done = False
        while (done is False):
            next_s, r, done = ds.step(curr_s)
            G += math.pow(gamma, power) * r
            power += 1
            curr_s = next_s
        # end while
        G_history.append(G)
    # end for
    v = np.mean(G_history)
    return v, G_history


def Sampling(ds, start_state, episodes, gamma):
    G_mean = 0

    if (start_state in ds.end_states):
        # 最后一个状态也可能有reward值
        return ds.R[start_state.value]
    
    for episode in tqdm.trange(episodes):
        curr_s = start_state
        G = ds.get_reward(curr_s)
        power = 1
        done = False
        while (done is False):
            next_s, r, done = ds.step(curr_s)
            G += math.pow(gamma, power) * r
            power += 1
            curr_s = next_s
        # end while
        G_mean += G
    # end for
    v = G_mean / episodes
    return v


if __name__=="__main__":
    episodes = 10000
    gamma = 1
    ds = data.Model()
    V = Sampling_MultiProcess(ds, episodes, gamma)
    for s in ds.S:
        print(str.format("{0}:{1}", s.name, V[s.value]))
