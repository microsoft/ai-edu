import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl


def normal_equation(X,Y):
    num_example = X.shape[0]
    # 在原始的X矩阵最左侧加一列1
    ones = np.ones((num_example,1))
    x = np.column_stack((ones, X))    
    # X^T * X
    p = np.dot(x.T, x)
    # (X^T * X)^{-1}
    #I = np.eye(p.shape[0]) * 1e-6
    #p = p + I
    q = np.linalg.inv(p)
    # (X^T * X)^{-1} * X^T
    r = np.dot(q, x.T)
    # (X^T * X)^{-1} * X^T * Y
    A = np.dot(r, Y)
    # 按顺序
    b = A[0,0]
    a1 = A[1,0]
    a2 = A[2,0]
    return a1, a2, b

def show_result(X, Y, a1, a2, b):
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False

    fig = plt.figure()
    plt.title(u"三维空间中最小二乘法的屏幕拟合结果")
    plt.axis('off')

    # 准备拟合平面数据
    axis_x = np.linspace(0,5, 11)
    axis_y = np.linspace(0,5, 11)
    P,Q = np.meshgrid(axis_x, axis_y)
    R = a1 * P + a2 * Q + b
    # 绘制拟合平面
    ax = fig.add_subplot(121,projection='3d')
    ax.plot_surface(P, Q, R, alpha=0.5)
    # 绘制原始样本点
    ax.scatter(X[:,0],X[:,1],Y, color='Red')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    # 绘制拟合平面
    ax = fig.add_subplot(122,projection='3d')
    ax.plot_surface(P, Q, R, alpha=0.3)
    # 绘制原始样本点
    ax.scatter(X[:,0],X[:,1],Y, color='black')
    # 绘制点到屏幕的竖线
    Y_hat = a1 * X[:,0] + a2 * X[:,1] + b
    for i in range(X.shape[0]):
        x = [X[i,0],X[i,0]]
        y = [X[i,1],X[i,1]]
        z = [Y_hat[i],Y[i,0]]
        print(x,y,z)
        ax.plot3D(x, y, z, color='Red')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    
    plt.show()

if __name__ == '__main__':
    X = np.array([[1,1],[2,3],[3,4],[4,3]])
    Y = np.array([5, 3, 4, 2]).reshape(4,1)
    a1, a2, b = -0.89, 0.13, 5.37
    print(str.format("a1={0:.2f}, a2={1:.2f}, b={2:.2f}", a1, a2, b))
    show_result(X, Y, a1, a2, b)
