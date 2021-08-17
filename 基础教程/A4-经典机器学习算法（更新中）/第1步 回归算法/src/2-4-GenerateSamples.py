# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import matplotlib as mpl

def generate_samples(a1,a2,b,n):
    # 模拟一个距离市中心的正态分布
    loc_R = (20+2)/2
    X1 = np.random.normal(loc=loc_R, scale=3.4, size=(n, 1))
    # 模拟一个房屋面积的正态分布
    loc_S = (40+120)/2
    X2 = np.random.normal(loc=loc_S, scale=11.2, size=(n, 1))
    
    Y = np.zeros_like(X1)
    for i in range(n):
        # 返回均值为0，方差为20的误差的一个值
        epsilon = np.random.normal(loc=0, scale=10, size=None)
        # 对于每个特定的x值，都从N(0,0.05)中取出一个随机值作为噪音添加到y上
        Y[i,0] = a1 * (20-X1[i,0]) + a2 * X2[i,0] + b + epsilon

    return X1, X2, Y

def generate_file_path(file_name):
    curr_path = sys.argv[0]
    curr_dir = os.path.dirname(curr_path)
    file_path = os.path.join(curr_dir, file_name)
    return file_path

def load_file(file_name):
    file_path = generate_file_path(file_name)
    file = Path(file_path)
    if file.exists():
        samples = np.loadtxt(file, delimiter=',')
        return samples
    else:
        return None

def show_sample(X1,X2,Y):
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    
    fig = plt.figure()
    plt.title(u"给定位置和面积的房价预测")
    plt.axis('off')
    # 绘制左视图
    ax = fig.add_subplot(121,projection='3d')
    ax.scatter(X1,X2,Y)
    ax.set_xlabel(u"距离")
    ax.set_ylabel(u"面积")
    ax.set_zlabel(u"价格")
    # 绘制右视图（然后手工把右视图的角度调整一下，与左侧对比）
    ax = fig.add_subplot(122,projection='3d')
    ax.scatter(X1,X2,Y)
    ax.set_xlabel(u"距离")
    ax.set_ylabel(u"面积")
    ax.set_zlabel(u"价格")

    plt.show()

if __name__ == '__main__':
   
    file_name = "2-0-data.csv"
    samples = load_file(file_name)
    if (samples is not None):
        X1 = samples[:, 0]
        X2 = samples[:, 1]
        Y = samples[:, 2]        
    else:
        a1 = 2
        a2 = 5
        b = 10
        n = 500
        X1,X2,Y=generate_samples(a1, a2, b, n)
        file_path = generate_file_path(file_name)
        samples = np.hstack((X1,X2,Y))
        np.savetxt(file_path, samples, fmt='%f, %f, %f', delimiter=',', header='x1, x2, y')
    #endif
    print(str.format("距离：最小值={0:.2f}，最大值={1:.2f}，均值={2:.2f}", X1.min(), X1.max(), X1.mean()))
    print(str.format("面积：最小值={0:.2f}，最大值={1:.2f}，均值={2:.2f}", X2.min(), X2.max(), X2.mean()))
    show_sample(X1,X2,Y)

    