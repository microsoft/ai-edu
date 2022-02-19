import numpy as np
import Data_Random_Walker as ds
import Algorithm_MC as algoMC

if __name__=="__main__":
    alpha = 0.01
    gamma = 1
    epsiodes = 1000
    V = np.zeros(7)
    V[1:6] = 0.5
    v = algoMC.MC(V, ds, epsiodes, alpha, gamma)
    print(v*6)
    print(np.around(v*6, 2))
