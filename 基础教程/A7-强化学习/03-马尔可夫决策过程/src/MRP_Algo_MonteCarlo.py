import math
import numpy as np
import tqdm
import multiprocessing as mp

def single_process(Rewards, TransMatrix, States, start_state, end_states, episodes, gamma):
    num_state = len(Rewards)
    sum_gain = 0
    for episode in tqdm.trange(episodes):
        if (start_state in end_states):
            # 最后一个状态也可能有reward值
            return Rewards[start_state.value]
        curr_state_value = start_state.value
        gain = Rewards[curr_state_value]
        power = 1
        while (True):
            next_state_value = np.random.choice(num_state, p=TransMatrix[curr_state_value])
            r = Rewards[next_state_value]
            gain += math.pow(gamma, power) * r
            if (States(next_state_value) in end_states):
                # 到达终点，分幕结束
                break
            else:
                power += 1
                curr_state_value = next_state_value
        # end while
        sum_gain += gain
    # end for
    v = sum_gain / episodes
    return v  

def run(Rewards, TransMatrix, States, end_states, gamma, episodes):
    pool = mp.Pool(processes=6)
    Vs = []
    results = []
    for start_state in States:
        results.append(pool.apply_async(single_process, args=(Rewards, TransMatrix, States, start_state, end_states, episodes, gamma,)))
    pool.close()
    pool.join()
    for i in range(len(results)):
        v = results[i].get()
        Vs.append(v)

    return Vs
