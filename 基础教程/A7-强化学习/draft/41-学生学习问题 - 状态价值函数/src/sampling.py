
import tqdm
import multiprocessing as mp
import math
import numpy as np
import StudentData as data


def Sampling_MultiProcess(ds, episodes, gamma):
    pool = mp.Pool(processes=4)
    V = np.zeros((ds.num_states))
    results_v = []
    g_s = []
    for start_state in ds.S:
        results_v.append(pool.apply_async(Sampling, 
                args=(ds, start_state, episodes, gamma,)
            )
        )
    pool.close()
    pool.join()
    for i in range(len(results_v)):
        v, g_per_episode = results_v[i].get()
        V[i] = v
        g_s.append(g_per_episode)

    return V, g_s


def RMSE(a, b):
    err = np.sqrt(np.sum(np.square(a-b))/a.shape[0])
    return err


def Sampling(ds, start_state, episodes, gamma):
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


if __name__=="__main__":
    episodes = 100
    gamma = 1
    ds = data.Data()
    V, g = Sampling_MultiProcess(ds, episodes, gamma)
    for s in ds.S:
        print(str.format("{0}:{1}", s.name, V[s.value]))

    G = np.array(g)
    print(G.shape)