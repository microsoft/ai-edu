import numpy as np
import Data_Random_Walker as ds
import Algorithm_Model_Free_Prediction as algo
import Algorithm_MRP as algoMRP

if __name__=="__main__":

    gamma = 1
    v = algoMRP.Matrix(ds, gamma)
    print(v*6)

    algoMRP.MonteCarol

    alpha = 0.1
    epsiodes = 1000
    V = np.zeros(7)
    V[1:6] = 0.5

    v = algo.MC(V, ds, ds.States.RoadC, epsiodes, alpha, gamma)
    print(v*6)
    print(np.around(v*6, 2))

    v = algo.TD(V, ds, ds.States.RoadC, epsiodes, alpha, gamma)
    print(v*6)
    print(np.around(v*6, 2))
