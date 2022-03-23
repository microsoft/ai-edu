from concurrent.futures import process
import numpy as np
import Data_FrozenLake as ds
import Algorithm_MRP as algoMRP
import Algorithm_MC as algoMCE
import matplotlib.pyplot as plt
import multiprocessing as mp
import Algorithm_TD as algoTD
import tqdm

def MultipleProcess(repeat, algo_fun, ds, start_state, episodes, alpha, gamma, ground_truth, every_n_episode):
    print(algo_fun)
    pool = mp.Pool(processes=8)
    Errors = []
    Values = []
    results = []
    for i in range(repeat):
        results.append(
            pool.apply_async(
                algo_fun, args=(
                    ds, start_state, episodes, alpha, gamma, ground_truth, every_n_episode,
                )
            )
        )

    pool.close()
    pool.join()
    for i in range(len(results)):
        value, error = results[i].get()
        Values.append(value)
        Errors.append(error)

    #print("每次运行的最终结果:")
    #print(Values)

    # use to draw plot
    mean_errors = np.mean(np.array(Errors), axis=0) 
    # 多次运行的平均值
    final_mean_value = np.mean(np.array(Values), axis=0)
    print("多次运行的平均 V 值:")
    print(np.round(final_mean_value,2).reshape(4,4))

    # 多次运行的平均值误差
    final_mean_value_error = RMSE(final_mean_value, ground_truth)
    print("多次运行的平均值的误差:", final_mean_value_error)

    # 得到多次运行的最终的误差
    final_errors = np.array(Errors)[:,-1]
    # 最终的误差值的平均值
    final_mean_error = np.mean(final_errors)
    # 最终的误差值的方差
    final_var_error = np.var(final_errors)
    print("多次运行的每次误差的平均值:", final_mean_error)
    print("多次运行的每次误差的方差:", final_var_error)

    return mean_errors


def MultipleProcess_all(repeat, algo_fun, ds, start_state, episodes, alpha, gamma, ground_truth, every_n_episode):
    print(algo_fun)
    pool = mp.Pool(processes=8)
    Errors1 = []
    Values1 = []
    Errors2 = []
    Values2 = []
    Errors3 = []
    Values3 = []
    Errors4 = []
    Values4 = []
    results = []
    for i in range(repeat):
        results.append(
            pool.apply_async(
                algo_fun, args=(
                    ds, start_state, episodes, alpha, gamma, ground_truth, every_n_episode,
                )
            )
        )

    pool.close()
    pool.join()
    for i in range(len(results)):
        V1, error1, V2, error2, V3, error3, V4, error4 = results[i].get()
        Values1.append(V1)
        Values2.append(V2)
        Values3.append(V3)
        Values4.append(V4)
        Errors1.append(error1)
        Errors2.append(error2)
        Errors3.append(error3)
        Errors4.append(error4)

    mean_errors1 = process_result(Values1, Errors1)
    mean_errors2 = process_result(Values2, Errors2)
    mean_errors3 = process_result(Values3, Errors3)
    mean_errors4 = process_result(Values4, Errors4)
    return mean_errors1, mean_errors2, mean_errors3, mean_errors4

def process_result(Values, Errors):
    # use to draw plot
    mean_errors = np.mean(np.array(Errors), axis=0) 
    # 多次运行的平均值
    final_mean_value = np.mean(np.array(Values), axis=0)
    print("多次运行的平均 V 值:")
    print(np.round(final_mean_value,2).reshape(4,4))

    # 多次运行的平均值误差
    final_mean_value_error = RMSE(final_mean_value, ground_truth)
    print("多次运行的平均值的误差:", final_mean_value_error)

    # 得到多次运行的最终的误差
    final_errors = np.array(Errors)[:,-1]
    # 最终的误差值的平均值
    final_mean_error = np.mean(final_errors)
    # 最终的误差值的方差
    final_var_error = np.var(final_errors)
    print("多次运行误差的平均值:", final_mean_error)
    print("多次运行的误差的方差:", final_var_error)

    return mean_errors


def FrozenLake_Matrix(gamma):
    vs = algoMRP.Matrix(ds, gamma)
    print("Matrix as Ground Truth")
    print(np.round(np.array(vs).reshape(4,4), 2))
    return vs

def RMSE(a, b):
    err = np.sqrt(np.sum(np.square(a - b))/b.shape[0])
    return err

def set_end_state_value(v):
    v[2] = -1
    v[8] = -1
    v[10] = -1
    v[15] = 5
    return v

if __name__=="__main__":
    gamma = 0.9
    # for MC
    ground_truth1 = FrozenLake_Matrix(gamma)
    # for TD
    ground_truth2 = ground_truth1.copy()
    ground_truth2[2] = 0
    ground_truth2[8] = 0
    ground_truth2[10] = 0
    ground_truth2[15] = 0

    episodes = 4000
    repeat = 10
    checkpoint = 10
    alpha = 0.02
    
    mean_errors = MultipleProcess(repeat, algoMCE.MC2, ds.Data_Frozen_Lake(), ds.States.Start, episodes, alpha, gamma, ground_truth1, checkpoint)
    plt.plot(mean_errors, label="MC2 - full1")

    alphas = [0.01, 0.02]
    for alpha in alphas:
        mean_errors = MultipleProcess(repeat, algoTD.TD_0, ds.Data_Frozen_Lake(), None, episodes, alpha, gamma, ground_truth2, checkpoint)
        plt.plot(mean_errors, label="TD_0 - " + str(alpha))

        mean_errors = MultipleProcess(repeat, algoTD.TD_batch, ds.Data_Frozen_Lake(), None, episodes, alpha, gamma, ground_truth2, checkpoint)
        plt.plot(mean_errors, label="TD_batch - " + str(alpha), linestyle='--')


    '''
    v = algoTD.TD(ds.Data_Frozen_Lake(), ds.States.Start, episodes, alpha, gamma)
    print(np.round(v, 2).reshape(4,4))
    print(RMSE(v, ground_truth))
    '''
    plt.grid()
    plt.legend()
    plt.show()
