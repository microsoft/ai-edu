import numpy as np
import Data_Random_Walker as ds
import Algorithm_Model_Free_Prediction as algo
import Algorithm_MRP as algoMRP

def RSME(v):
    global ground_truth
    err = np.sqrt(np.sum(np.square(v - ground_truth[1:6]))/V.shape[0])
    print("RSME=",err)

def set_end_state_value(v):
    v[0] = 0
    v[6] = 1
    return v

if __name__=="__main__":

    gamma = 1
    global ground_truth
    ground_truth = algoMRP.Matrix(ds, gamma)
    print(ground_truth)

    alpha = 0.02
    epsiodes = 1000
    V = np.zeros(7)
    #V[1:6] = 0.5
    v = algo.MC(V, ds.Data_Random_Walker(), ds.States.RoadC, epsiodes, alpha, gamma)
    print("\nMC - all state - 1")
    print(v)
    #print(np.around(v*6, 2))
    v = set_end_state_value(v)
    RSME(v[1:6])

    V = np.zeros(7)
    #V[1:6] = 0
    v = algo.MC2(V, ds.Data_Random_Walker(), ds.States.RoadC, epsiodes, alpha, gamma)
    print("\nMC - all state - 2")
    print(v)
    #print(np.around(v*6, 2))
    v = set_end_state_value(v)
    RSME(v[1:6])

    alpha = 0.1
    V = np.zeros(7)
    #V[1:6] = 0.5
    v = algo.TD(V, ds.Data_Random_Walker(), ds.States.RoadC, epsiodes, alpha, gamma)
    print("\nTD")
    print(v)
    #print(np.around(v*6, 2))
    v = set_end_state_value(v)
    RSME(v[1:6])
