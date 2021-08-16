# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import matplotlib as mpl

def mse(Y, Y_hat):
    return np.sum((Y-Y_hat) * (Y-Y_hat))

def b(X,Y):
    Loss = []
    B = []
    for b in np.linspace(start=0, stop=2.3, num=50):
        Y_hat = 0.5 * X + b     # a值固定为0.5
        loss = mse(Y, Y_hat)
        Loss.append(loss)
        B.append(b)
    return B, Loss

def a(X,Y):
    Loss = []
    A = []
    for a in np.linspace(start=0.2, stop=0.9, num=50):
        Y_hat = a * X + 1   # b值固定为1
        loss = mse(Y, Y_hat)
        Loss.append(loss)
        A.append(a)
    return A, Loss

def ab(A, B, X, Y):
    R = np.zeros((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            Y_hat = A[i] * X + B[j]
            R[i,j] = mse(Y, Y_hat)
    return R

def get_min_pos(X,Y):
    min_value = min(Y)
    idx = Y.index(min_value)
    x = X[idx]
    y = Y[idx]
    return min_value, x, y

def get_min_pos_3d(X,Y,Z):
    min_value = np.min(Z)
    pos = np.argmin(Z)
    x = X[pos // Z.shape[0]]
    y = Y[pos % Z.shape[0]]
    return min_value, x, y

def show_sample(A, Loss1, B, Loss2, X, Y):
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    
    fig = plt.figure()
    plt.title(u"均方差损失函数的理解")
    plt.axis('off')

    # 绘制左上视图
    ax = fig.add_subplot(221)
    ax.scatter(A,Loss1,s=5)
    ax.set_title(u"参数b=1时")
    ax.set_xlabel(u"参数a的变化")
    ax.set_ylabel(u"损失函数")
    min_value, x, y = get_min_pos(A, Loss1)
    ax.text(x,y,str.format("最小值={0:.3f}",min_value), ha='center', va='bottom')

    # 绘制右上视图
    ax = fig.add_subplot(222)
    ax.scatter(B,Loss2,s=5)
    ax.set_title(u"参数a=0.5时")
    ax.set_xlabel(u"参数b的变化")
    ax.set_ylabel(u"损失函数")
    min_value, x, y = get_min_pos(B, Loss2)
    ax.text(x,y,str.format("最小值={0:.3f}",min_value), ha='center', va='bottom')

    # 绘制左下视图
    ax = fig.add_subplot(223, projection="3d")
    P,Q = np.meshgrid(A, B)
    R = ab(A,B,X,Y)
    ax.set_title(u"损失函数三维图")
    ax.set_xlabel(u"参数a的变化")
    ax.set_ylabel(u"参数b的变化")
    ax.set_zlabel(u"损失函数")
    ax.plot_surface(P, Q, R, cmap='rainbow')
    plt.contour(P,Q,R,levels=np.logspace(-5, 5, 50), norm=LogNorm(), cmap=plt.cm.jet)

    # 绘制右下视图
    ax = fig.add_subplot(224)
    ax.set_title(u"损失函数等高线图")
    ax.set_xlabel(u"参数a的变化")
    ax.set_ylabel(u"参数b的变化")
    plt.contour(P,Q,R,levels=np.logspace(-5, 5, 50), norm=LogNorm(), cmap=plt.cm.jet)
    ax.text(0,0,str.format("最小值={0:.3f}",np.min(R)))
    min_value, x, y = get_min_pos_3d(A,B,R)
    ax.text(x,y,str.format("最小值={0:.3f}",min_value), ha='center', va='top')

    plt.show()

if __name__ == '__main__':
    X = np.array([2,3,4])
    Y = np.array([2,3,3])
    B, LossB = b(X,Y)
    A, LossA = a(X,Y)
    R = ab(A,B,X,Y)

    show_sample(A, LossA, B, LossB, X, Y)
