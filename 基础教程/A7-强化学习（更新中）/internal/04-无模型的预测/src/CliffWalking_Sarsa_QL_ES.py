import Algorithm_TD as algoTD
import Data_CliffWalking as data_cliff
import multiprocessing as mp
import numpy as np

def MultipleProcess(repeat, algo_fun, ds, start_state, episodes, alpha, gamma, checkpoint):
    print(algo_fun)
    pool = mp.Pool(processes=8)
    Errors = []
    Values = []
    results = []
    for i in range(repeat):
        results.append(
            pool.apply_async(
                algo_fun, args=(
                    ds, start_state, episodes, alpha, gamma, checkpoint,
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
    #final_mean_value_error = RMSE(final_mean_value, ground_truth)
    #print("多次运行的平均值的误差:", final_mean_value_error)

    # 得到多次运行的最终的误差
    final_errors = np.array(Errors)[:,-1]
    # 最终的误差值的平均值
    final_mean_error = np.mean(final_errors)
    # 最终的误差值的方差
    final_var_error = np.var(final_errors)
    print("多次运行的每次误差的平均值:", final_mean_error)
    print("多次运行的每次误差的方差:", final_var_error)

    return mean_errors

import Data_FrozenLake2 as dfl2
import Data_CliffWalking as dc
import matplotlib.pyplot as plt


def draw_arrow(Q, width=6):
    np.set_printoptions(suppress=True)
    #print(np.round(Q, 3))
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            if (Q[i,j] == 0):
                Q[i,j]=-10

    chars = [0x2191, 0x2192, 0x2193, 0x2190]
    for i in range(Q.shape[0]):
        if np.sum(Q[i,:]) == -40:
            print("O", end="")
        else:
            idx = np.argmax(Q[i,:])
            print(chr(chars[idx]), end="")
        print(" ", end="")
        if ((i+1) % width == 0):
            print("")


if __name__=="__main__":
    env = dc.Env()
    episodes = 10000
    EPSILON = 0.1
    GAMMA = 1
    ALPHA = 0.1
    repeat = 1
    Errors1 = []
    Errors2 = []
    Errors3 = []
    for i in range(repeat):
        Q1, errors1 = algoTD.Sarsa(env, True, episodes, ALPHA, GAMMA, EPSILON, 2)
        Q2, errors2 = algoTD.Q_Learning(env, True, episodes, ALPHA, GAMMA, EPSILON, 2)
        Q3, errors3 = algoTD.E_Sarsa(env, True, episodes, ALPHA, GAMMA, EPSILON, 2)
        Errors1.append(errors1)
        Errors2.append(errors2)
        Errors3.append(errors3)
    
    mean_errors1 = np.mean(np.array(Errors1), axis=0) 
    mean_errors2 = np.mean(np.array(Errors2), axis=0) 
    mean_errors3 = np.mean(np.array(Errors3), axis=0) 
    
    print("-"*20)
    print("Saras")
    draw_arrow(Q1, width=6)
    print("-"*20)
    print("Q-learning")
    draw_arrow(Q2, width=6)
    print("-"*20)
    print("E-Sarsa")
    draw_arrow(Q3, width=6)
    exit(0)
    plt.plot(mean_errors1, label="Saras")
    plt.plot(mean_errors2, label="Q-le")
    plt.plot(mean_errors3, label="E-Saras")
    plt.legend()
    plt.show()
