
import gymnasium as gym
import numpy as np
import tqdm
from common.Algo_TD_Base import TD_Base
import matplotlib.pyplot as plt
import math
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
mpl.rcParams['axes.unicode_minus']=False


if __name__=="__main__":
    lambd = 0.95
    decade = []
    for i in range(50):
        value = (1-lambd) * math.pow(lambd, i)
        decade.append(value)
        plt.plot([i, i], [0, value], color="black", linestyle="--", linewidth=1)
    plt.plot(decade, marker="o")
    plt.grid()
    plt.show()
