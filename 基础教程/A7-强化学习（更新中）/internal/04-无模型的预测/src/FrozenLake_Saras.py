import numpy as np
import Algorithm_TD as algoTD


import Data_FrozenLake2 as dfl2

if __name__=="__main__":

    env = dfl2.Data_FrozenLake_Env()
    Q,_ = algoTD.Sarsa(env, False, 10000, 0.01, 0.9, None, 10)
    np.set_printoptions(suppress=True)
    print(np.round(Q, 3))
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            if (Q[i,j] == 0):
                Q[i,j]=-10

    chars = [0x2190, 0x2191, 0x2192, 0x2193]
    for i in range(Q.shape[0]):
        if np.sum(Q[i,:]) == -40:
            print("O", end="")
        else:
            idx = np.argmax(Q[i,:])
            print(chr(chars[idx]), end="")
        print(" ", end="")
        if ((i+1) % 4 == 0):
            print("")


