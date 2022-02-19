import numpy as np


def MC(V, ds, episodes, alpha, gamma):
    for i in range(episodes):
        trajectory = []
        curr_state = ds.States.RoadC
        trajectory.append((curr_state.value, ds.get_reward(curr_state)))
        while True:
            # 左右随机游走
            next_state = ds.get_random_next_state(curr_state)
            #endif
            reward = ds.get_reward(next_state)
            trajectory.append((next_state.value, reward))
            curr_state = next_state
            # 到达终点，结束一幕，退出循环开始算分
            if (ds.is_end_state(curr_state)):
                break
            #endif
        #endwhile
        # calculate G,V
        G = 0
        # 从后向前遍历，因为 G = R_t+1 + gamma * R_t + gamme^2 * R_t-1 + ...
        for j in range(len(trajectory)-1, -1, -1):
            state_value, reward = trajectory[j]
            G = gamma * G + reward
        
        # 只更新起始状态的V值，中间的都忽略
        #s,r = trajectory[0]
        #V1[s] = V1[s] + alpha * (G - V1[s])

        # 更新从状态开始到终止状态之前的所有V值
        for (state_value, reward) in trajectory[0:-1]:
            # math: V(s) \leftarrow V(s) + \alpha (G - V(s))
            V[state_value] = V[state_value] + alpha * (G - V[state_value])
    #endfor
    return V
