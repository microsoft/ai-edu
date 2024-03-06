
import gymnasium as gym
import numpy as np
import tqdm
from common.Algo_TD_Base import TD_Base
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
mpl.rcParams['axes.unicode_minus']=False


if __name__=="__main__":
    trace = [0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0]
    gamma = 0.9
    alpha = 0.1
    lambd = 0.8
    grid = plt.GridSpec(nrows=3, ncols=1)
    betas = [1, 1-alpha, 0]
    titles = [r"$\beta=1$ 累积迹", r"$\beta=1-\alpha$ 荷兰迹", r"$\beta=0$ 替代迹"]
    for row, beta in enumerate(betas):
        ets = [0]
        for i in range(1, len(trace)):
            if trace[i] == 0:
                et = ets[i-1] * gamma * lambd
            else:
                et = 1 + ets[i-1] * gamma * lambd * beta
            ets.append(et)
        print(ets)
        plt.subplot(grid[row, 0])
        plt.plot(ets)
        plt.title(titles[row])
        plt.grid()
    plt.show()
