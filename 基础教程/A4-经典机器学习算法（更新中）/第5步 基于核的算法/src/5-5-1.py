from matplotlib.colors import LogNorm
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.core.fromnumeric import argmin

def draw_surface(ax, a2, a3, flag):
    P,Q = np.meshgrid(a2, a3)
    R = 4*P*P + 13*Q*Q/2 + 10*P*Q -2*P - 2*Q
    z = np.min(R)
    idx = np.argmin(R)
    y = a2[idx // 1000]
    x = a3[idx % 1000]
    ax.scatter(x,y,z)
    print(str.format("{3}: a2={0:.2f}, a3={1:.2f}, d*={2:.2f}", x,y,z, flag))
    ax.plot_surface(P, Q, R, alpha=0.5)

# 统计学习方法中的例子图示
def test():
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()

    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlabel("a2")
    ax.set_ylabel("a3")
    ax.set_title("(-2,2)取值空间")
    a2 = np.linspace(-2,2,1000)
    a3 = np.linspace(-2,2,1000)
    draw_surface(ax,a2,a3, "left")

    ax = fig.add_subplot(122, projection='3d')
    ax.set_title("(0,0.3)取值空间")
    a2 = np.linspace(0,0.3,1000)
    a3 = np.linspace(0,0.3,1000)
    draw_surface(ax,a2,a3, "right")
    ax.set_xlabel("a2")
    ax.set_ylabel("a3")

    plt.show()
 
if __name__ == '__main__':
    test()
