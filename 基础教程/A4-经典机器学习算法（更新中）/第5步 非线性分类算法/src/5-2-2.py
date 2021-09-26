from matplotlib.colors import LogNorm
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.core.fromnumeric import argmin

def draw_x_y_1(ax):
    # 绘制x+y-1=0平面，垂直于x-y 平面
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    P = X
    Q = -X+1
    R = 5*np.maximum(Y,0) # 增加平面的高度
    s1 = ax.plot_surface(P, Q, R, color='y', alpha=0.6)
    
    #两平面交线
    x = np.linspace(-2, 3, 100)
    y = -x+1
    z = 2 * np.square(x) - 2 * x + 1
    ax.plot3D(x, y, z, color='r', alpha=0.6, label='x+y-1=0', linestyle='--')
    
    # 极值点
    x = 0.5
    y = 0.5
    z = 0.5
    ax.scatter(x, y, z, c='r', marker='*', s=50, label='$p_1(0.5,0.5,0,5)$')

def draw_x_y_2(ax):
    # 绘制x+y+2=0平面，垂直于x-y 平面
    x = np.linspace(-3, 1, 100)
    y = np.linspace(-3, 1, 100)
    X, Y = np.meshgrid(x, y)
    P = X
    Q = -X-2
    R = 20*np.maximum(Y,0) # 增加平面的高度
    ax.plot_surface(P, Q, R, color='y', alpha=0.6)
    
    # 两平面交线
    y = -x-2
    z = 2 * np.square(x) + 4 * x + 4
    ax.plot3D(x, y, z, color='r', alpha=0.6, label='x+y+2=0', linestyle='-.')
    
    # 极值点
    x = -1
    y = -1
    z = 2
    ax.scatter(x, y, z, c='r', marker='^', s=50, label='$p_2(-1,-1,2)$')
   

def draw_x2_y2(ax):
    # 绘制 x^2+y^2 曲面
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    P, Q = np.meshgrid(x, y)
    R = P * P + Q * Q
    ax.plot_surface(P, Q, R, alpha=0.1)
    #ax.contour(P,Q,R, levels=np.linspace(-10, 10, 11)) 
    # 极值点
    x = 0
    y = 0
    z = 0
    ax.scatter(x, y, z, c='r', marker='o', s=50, label='$p_0(0,0,0)$')
    return P,Q,R

def draw_2D(ax2,P,Q,R):
    c = ax2.contour(P,Q,R, levels=np.linspace(-10, 10, 11))
    ax2.clabel(c,inline=1,fontsize=10)
    ax2.grid()
    
    x = np.linspace(-3, 1, 1000)
    y = -x - 2
    ax2.plot(x,y,label='x+y+2=0',linestyle='-.')
    x = np.linspace(-2, 3, 1000)
    y = -x+1
    ax2.plot(x,y,label='x+y-1=0',linestyle='--')
    ax2.legend()
    ax2.scatter(0, 0, marker='o',s=50, label='$p_0(0,0,0)$')
    ax2.scatter(0.5, 0.5, marker='*',s=50, label='$p_1(0.5,0.5,0.5)$')
    ax2.scatter(-1, -1, marker='^',s=50, label='$p_2(-1,-1,2)$')

def draw_x2y2_xy12():
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("三维图")

    P,Q,R = draw_x2_y2(ax)
    draw_x_y_2(ax)
    draw_x_y_1(ax)
    ax.legend()

    ax2 = fig.add_subplot(122)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("二维投影图")
    draw_2D(ax2,P,Q,R)
    ax2.legend()

    plt.show()

if __name__ == '__main__':
    draw_x2y2_xy12()
