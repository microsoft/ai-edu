# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import matplotlib as mpl

# 错误的代码
def generate_samples_1(a, b, m):
    # m个[0,1)之间随机数，表示机房内计算机数量/1000
    X = np.random.random(size=(m, 1))
    # 返回均值为0，方差为0.1的误差的一组值
    Epsilon = np.random.normal(loc=0, scale=0.05, size=X.shape)
    Y = a * X + b + Epsilon
    return X,Y

# 正确的代码
def generate_samples_2(a, b, m):
    # 以0.5为中心的正态分布，表示机房内计算机数量/1000
    X = np.random.normal(loc=0.5, scale=0.12, size=(m, 1))
    Y = np.zeros_like(X)
    for i in range(m):
        # 返回均值为0，方差为0.05的误差的一个值
        epsilon = np.random.normal(loc=0, scale=0.05, size=None)
        # 对于每个特定的x值，都从N(0,0.05)中取出一个随机值作为噪音添加到y上
        Y[i,0] = a * X[i,0] + b + epsilon
    return X,Y

def generate_file_path(file_name):
    curr_path = sys.argv[0]
    curr_dir = os.path.dirname(curr_path)
    file_path = os.path.join(curr_dir, file_name)
    return file_path

def show_sample(X,Y):
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    
    plt.figure()

    plt.subplot(212)
    plt.scatter(X,Y,s=10)
    plt.title(u"机房空调功率预测")
    plt.xlabel(u"服务器数量(千台)")
    plt.ylabel(u"空调功率(千瓦)")
    
    plt.subplot(221)
    plt.title(u"X的分布")
    plt.hist(X)
    plt.subplot(222)
    plt.title(u"Y的分布")
    plt.hist(Y)

    plt.show()

if __name__ == '__main__':
    file_name = "1-0-data.csv"
    file_path = generate_file_path(file_name)
    file = Path(file_path)
    if file.exists():
        samples = np.loadtxt(file, delimiter=',')
        X = samples[:, 0]
        Y = samples[:, 1]
    else:
        a = 0.5         # 参数a
        b = 1           # 参数b
        m = 200         # 模拟200个机房的样本
        X,Y = generate_samples_2(a, b, m)
        samples = np.hstack((X,Y))
        np.savetxt(file_path, samples, fmt='%f, %f', delimiter=',', header='x, y')
    #endif
    show_sample(X,Y)
    