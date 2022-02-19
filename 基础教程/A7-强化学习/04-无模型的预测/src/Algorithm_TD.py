import numpy as np


def TD(V, ds, episodes, alpha, gamma):
    for i in range(episodes):
        curr_state = ds.States.RoadC
        while True:
            # 左右随机游走
            next_state = ds.get_random_next_state(curr_state)
            #endif
            reward = ds.get_reward(next_state)
            # 立刻更新状态值，不等本幕结束
            # math: V(S) \leftarrow V(S) + \alpha[R + \gamma V(S') - V(S)]
            V[curr_state.value] = V[curr_state.value] + alpha * (reward + gamma * V[next_state.value] - V[curr_state.value])
            curr_state = next_state
            # 到达终点，结束一幕
            if (ds.is_end_state(curr_state)):
                break
            #endif
        #endwhile
    #endfor
    return V
