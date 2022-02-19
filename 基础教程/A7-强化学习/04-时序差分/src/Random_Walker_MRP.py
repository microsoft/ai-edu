import numpy as np
import Random_Walker_MRP_baseline as ds


def MC(ds, episodes, alpha):
    V1 = np.zeros(7)
    V1[1:6] = 0.5
    V2 = np.zeros(7)
    V2[1:6] = 0.5

    for i in range(episodes):
        trajectory = []
        state = 3
        trajectory.append((state,0))
        while True:
            # 左右随机游走
            if (np.random.binomial(1, 0.5) == Action.MOVE_LEFT.value):
                next_state = state - 1
            else:
                next_state = state + 1
            #endif
            R = 0
            if (next_state == 6):
                R = 1
            #endif
            trajectory.append((next_state, R))
            state = next_state
            if (state == 0 or state == 6):
                break
            #endif
        #endwhile
        # calculate G,V
        gamma = 1
        G = 0
        for j in range(len(trajectory)-1, -1, -1):
            s,r = trajectory[j]
            G = gamma * G + r
        
        # 只更新起始状态的V值，中间的都忽略
        #s,r = trajectory[0]
        #V1[s] = V1[s] + alpha * (G - V1[s])

        # 更新从状态开始到终止状态之前的所有V值
        for (s,r) in trajectory[0:-1]:
            # math: V(s) \leftarrow V(s) + \alpha (G - V(s))
            V2[s] = V2[s] + alpha * (G - V2[s])
    #endfor
    #print(V1*6)
    print(V2*6)
    return V2


if __name__=="__main__":
    alpha = 1
    MC(ds, alpha)
