from concurrent.futures import process
import numpy as np
import Data_A_B as ds
import Algorithm_MRP as algoMRP
import Algorithm_MC as algoMCE
import Algorithm_TD as algoTD
import matplotlib.pyplot as plt
import multiprocessing as mp
import Algorithm_TD as algoTD
import tqdm


def RMSE(a, b):
    err = np.sqrt(np.sum(np.square(a - b))/b.shape[0])
    return err


if __name__=="__main__":
    gamma = 1
    ground_truth1 = algoMRP.Matrix(ds, gamma)
    print(ground_truth1)
    print(algoMRP.Bellman(ds, gamma))
    v,_ = algoMCE.MC2(ds.A_B_Question(), ds.States.A, 10000, 0.01, 1, ground_truth1, 1)
    print("MC2=", v)
    v,_ = algoTD.TD0(ds.A_B_Question(), ds.States.A, 10000, 0.01, 1, ground_truth1, 1)
    print("TD0=", v)
    v,_ = algoTD.TD_batch(ds.A_B_Question(), ds.States.A, 10000, 0.01, 1, ground_truth1, 10)
    print("TD batch=", v)
