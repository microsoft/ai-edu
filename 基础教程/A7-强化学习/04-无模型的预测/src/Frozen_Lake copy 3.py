import numpy as np
import Data_FrozenLake as ds
import Algorithm_MRP as algoMRP
import Algorithm_MC as algoMC

def FrozenLake_Matrix(gamma):
    vs = algoMRP.Matrix(ds, gamma)
    print("Matrix")
    print(np.round(np.array(vs).reshape(4,4), 2))
    return vs

def FrozenLake_Bellman(gamma):
    vs = algoMRP.Bellman(ds, gamma)
    np.set_printoptions(suppress=True)
    print("Bellman")
    print(np.round(np.array(vs).reshape(4,4), 2))

def FrozenLake_MC1(repeat, episodes, gamma, ground_truth):
    print("MC1")
    errors = []
    for i in range(repeat):
        v = algoMC.MC1(ds.Data_Frozen_Lake(), episodes, gamma)
        print(np.round(np.array(v).reshape(4,4), 2))
        err = RMSE(v, ground_truth)
        errors.append(err)
    measure(errors)

def FrozenLake_MC2(repeat, episodes, gamma, ground_truth):
    print("MC2")
    errors = []
    for i in range(repeat):
        v = algoMC.MC2(ds.Data_Frozen_Lake(), ds.States.Start, episodes, gamma)
        print(np.round(np.array(v).reshape(4,4), 2))
        err = RMSE(v, ground_truth)
        errors.append(err)
    measure(errors)
    

def FrozenLake_MC3(repeat, episodes, alpha, gamma, ground_truth):
    print("MC3")
    err = 0
    for i in range(repeat):
        V = np.zeros(16)
        v = algoMC.MC3(V, ds.Data_Frozen_Lake(), ds.States.Start, episodes, alpha, gamma)
        print(np.round(np.array(v).reshape(4,4), 2))
        err += RMSE(v, ground_truth)
    print("average RMSE=", err/repeat)

def FrozenLake_MC4(repeat, episodes, alpha, gamma, ground_truth):
    print("MC4")
    for i in range(repeat):
        V = np.zeros(16)
        v = algoMC.MC4(V, ds.Data_Frozen_Lake(), ds.States.Start, episodes, alpha, gamma)
        print(np.round(np.array(v).reshape(4,4), 2))
        RMSE(v, ground_truth)

def measure(errors):
    print("----------------")
    print(errors)
    print(str.format("var={0}", np.var(errors)))
    print(str.format("mean={0}", np.mean(errors)))
    print(str.format("max={0}", np.max(errors)))
    print(str.format("min={0}", np.min(errors)))
    print("----------------")


def RMSE(a, b):
    err = np.sqrt(np.sum(np.square(a - b))/b.shape[0])
    print("RMSE=",err)
    return err

def set_end_state_value(v):
    v[2] = -1
    v[8] = -1
    v[10] = -1
    v[15] = 5
    return v

if __name__=="__main__":
    gamma = 0.9
    ground_truth = FrozenLake_Matrix(gamma)
    FrozenLake_Bellman(gamma)
    '''
    VV = np.zeros(16)
    RMSE(VV, ground_truth)
    exit(0)
    '''
    episodes = 10000
    repeat = 10
    FrozenLake_MC1(repeat, episodes, gamma, ground_truth)
    FrozenLake_MC2(repeat, episodes, gamma, ground_truth)
    exit(0)

    alpha = 0.01
  

    '''
    VV = np.zeros(16)
    for i in range(10):
        print("\nMC2" + str(i))
        V = np.zeros(16)
        v = algoMC.MC2(V, ds.Data_Frozen_Lake(), ds.States.Start, episodes, gamma)
        print(np.round(np.array(v).reshape(4,4), 2))
        v = set_end_state_value(v)
        RMSE(v, ground_truth)
        VV += v

    print(np.round(np.array(VV/10).reshape(4,4), 2))
    RMSE(VV/10, ground_truth)


    alphas = [0.01,0.02,0.03,0.05]
    errors = []
    for alpha in alphas:
        error = 0
        for i in range(10):
            print("\nMC3-" + str(alpha))
            V = np.zeros(16)
            v = algoMC.MC3(V, ds.Data_Frozen_Lake(), ds.States.Start, episodes, alpha, gamma)
            #print(np.round(np.array(v).reshape(4,4), 2))
            #v = set_end_state_value(v)
            error += RMSE(v, ground_truth)
        print(error)
        errors.append(error/10)
    print(errors)
    '''


    
    v = algoMC.MC4(V, ds.Data_Frozen_Lake(), ds.States.Start, episodes, alpha, gamma)
    print("\nMC4")
    print(np.round(np.array(v).reshape(4,4), 2))
    #set_end_state_value(v)
    RMSE(v, ground_truth)
    

    '''
    V = np.zeros(16)
    v = algo.TD(V, ds.Data_Frozen_Lake(), ds.States.Start, episodes, alpha, gamma)
    print("\nTD")
    print(np.round(np.array(v).reshape(4,4), 2))
    v = set_end_state_value(v)
    RMSE(v, ground_truth)
    '''
