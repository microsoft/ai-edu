import tqdm
import multiprocessing as mp
import math
import numpy as np




def TD(V, ds, start_state, episodes, alpha, gamma):
    for i in range(episodes):
        curr_state = start_state
        while True:
            # 到达终点，结束一幕
            if (ds.is_end_state(curr_state)):
                break
            # 随机游走
            next_state, reward = ds.step(curr_state)
            #endif
            # 立刻更新状态值，不等本幕结束
            # math: V(S) \leftarrow V(S) + \alpha[R + \gamma V(S') - V(S)]
            V[curr_state.value] = V[curr_state.value] + alpha * (reward + gamma * V[next_state.value] - V[curr_state.value])
            curr_state = next_state
            #endif
        #endwhile
    #endfor
    return V
