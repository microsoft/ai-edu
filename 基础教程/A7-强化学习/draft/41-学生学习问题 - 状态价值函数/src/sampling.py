
import tqdm
import multiprocessing as mp
import math
import numpy as np

import StudentData as data

'''
def MonteCarol(ds, end_states, gamma, episodes):
    pool = mp.Pool(processes=6)
    Vs = []
    results = []
    for start_state in ds.States:
        results.append(pool.apply_async(mc_single_process, 
        args=(ds.Rewards, ds.TransMatrix, ds.States, start_state, end_states, episodes, gamma,)))
    pool.close()
    pool.join()
    for i in range(len(results)):
        v = results[i].get()
        Vs.append(v)

    return Vs
'''

def Sampling(ds, start_state, episodes, gamma):
    sum_gain = 0
    for episode in tqdm.trange(episodes):
        if (start_state in ds.end_states):
            # 最后一个状态也可能有reward值
            return ds.R[start_state.value]
        curr_s = start_state
        gain = ds.R[curr_s.value]
        power = 1
        done = False
        while (done is False):
            next_s, r, done = ds.step(curr_s)
            gain += math.pow(gamma, power) * r
            power += 1
            curr_s = next_s
        # end while
        sum_gain += gain
    # end for
    v = sum_gain / episodes
    return v  

if __name__=="__main__":
    episodes = 10000
    gamma = 1
    ds = data.Data()
    
    print("----Monte Carol----")
    for start_state in ds.S:
        v = Sampling(ds, start_state, episodes, gamma)
        print(start_state, v)
