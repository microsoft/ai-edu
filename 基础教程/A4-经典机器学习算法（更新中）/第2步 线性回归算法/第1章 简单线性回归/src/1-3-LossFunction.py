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

# 参数a固定为0.5，遍历参数b=(start=0, stop=2.3, num=50)
def traversal_b(X,Y,scope_b):
    loss_b = []
    value_b = []
    for b in np.linspace(start=scope_b[0], stop=scope_b[1], num=scope_b[2]):
        Y_hat = 0.5 * X + b     # a值固定为0.5
        loss = mse(Y, Y_hat)    # 计算均方差损失函数
        loss_b.append(loss)     # 保存好便于后面绘图
        value_b.append(b)       # 保存好便于后面绘图
    return value_b, loss_b

# 参数b固定为1，遍历参数a=(start=0.2, stop=0.9, num=50)
def traversal_a(X,Y,scope_a):
    loss_a = []
    value_a = []
    for a in np.linspace(start=scope_a[0], stop=scope_a[1], num=scope_a[2]):
        Y_hat = a * X + 1       # b值固定为1
        loss = mse(Y, Y_hat)    # 计算均方差损失函数
        loss_a.append(loss)     # 保存好便于后面绘图
        value_a.append(a)       # 保存好便于后面绘图
    return value_a, loss_a

# 遍历参数a=[0.2, 0.9, 50] 同时 遍历参数b=[0, 2.3, 50]
def traversal_ab(X, Y, value_a, value_b):
    R = np.zeros((len(value_a), len(value_b)))
    # 遍历 a 和 b 的组合
    for i in range(len(value_a)):
        for j in range(len(value_b)):
            Y_hat = value_a[i] * X + value_b[j] # 计算回归值
            loss = mse(Y, Y_hat)                # 计算损失函数
            R[i,j] = loss                       # 保存好便于后面绘图
    return R

# 获得最小值的坐标
def get_min_pos(value, loss):
    min_value = min(loss)
    idx = loss.index(min_value)
    x = value[idx]
    y = value[idx]
    return min_value, x, y

# 在等高线图上获得最小值的坐标
def get_min_pos_3d(value_a, value_b, R):
    min_value = np.min(R)
    pos = np.argmin(R)
    x = value_a[pos // R.shape[0]]
    y = value_b[pos % R.shape[0]]
    return min_value, x, y

def show_sample(value_a, loss_a, value_b, loss_b, R):
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    
    fig = plt.figure()
    plt.title(u"均方差损失函数的理解")
    plt.axis('off')

    # 绘制左上视图
    ax = fig.add_subplot(221)
    ax.scatter(value_a, loss_a, s=5)
    ax.set_title(u"参数b=1时")
    ax.set_xlabel(u"参数a的变化")
    ax.set_ylabel(u"损失函数")
    min_value, x, y = get_min_pos(value_a, loss_a)
    ax.text(x,y,str.format("最小值={0:.3f}",min_value), ha='left', va='bottom')

    # 绘制右上视图
    ax = fig.add_subplot(222)
    ax.scatter(value_b, loss_b, s=5)
    ax.set_title(u"参数a=0.5时")
    ax.set_xlabel(u"参数b的变化")
    ax.set_ylabel(u"损失函数")
    min_value, x, y = get_min_pos(value_b, loss_b)
    ax.text(x,y,str.format("最小值={0:.3f}",min_value), ha='left', va='bottom')

    # 绘制左下视图
    ax = fig.add_subplot(223, projection="3d")
    P,Q = np.meshgrid(value_a, value_b)
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
    min_value, x, y = get_min_pos_3d(value_a, value_b, R)
    ax.text(x,y,str.format("最小值={0:.3f}",min_value), ha='center', va='top')

    plt.show()

if __name__ == '__main__':
    X = np.array([2,3,4])
    Y = np.array([2,3,3])
    scope_b = (0,2.3,50)
    value_b, loss_b = traversal_b(X, Y, scope_b)
    scope_a = (0.2,0.9,50)
    value_a, loss_a = traversal_a(X, Y, scope_a)
    R = traversal_ab(X, Y, value_a, value_b)

    show_sample(value_a, loss_a, value_b, loss_b, R)
