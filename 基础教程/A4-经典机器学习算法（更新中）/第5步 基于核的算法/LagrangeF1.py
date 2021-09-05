from matplotlib.colors import LogNorm
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.core.fromnumeric import argmin

def draw_x2y2_xy2():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 x^2+y^2 曲面
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    P, Q = np.meshgrid(x, y)
    R = P * P + Q * Q
    ax.plot_surface(P, Q, R, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # 绘制x+y+2=0平面，垂直于x-y 平面
    x = np.linspace(-3, 1, 100)
    y = np.linspace(-3, 1, 100)
    X, Y = np.meshgrid(x, y)
    P2 = X
    Q2 = -X-2
    R2 = 20*np.maximum(Y,0) # 增加平面的高度
    ax.plot_surface(P2, Q2, R2, color='y', alpha=0.6)

    #两平面交线
    x = np.linspace(-3, 1, 100)
    y = -x - 2
    z = 2 * np.square(x) + 4 * x + 4
    ax.plot3D(x, y, z, color='r', alpha=0.6)

    # 极值点
    x = -1
    y = -1
    z = 2
    ax.scatter(x, y, z, c='r')
    plt.show()

if __name__ == '__main__':
    draw_x2y2_xy2()
