# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_name = "../../data/ch05.npz"

# y = w1*x1 + w2*x2 + w3*x3 + b
# W1 = 朝向：1,2,3,4 = N,W,E,S
# W2 = 位置几环：2,3,4,5,6
# W3 = 面积:平米
def TargetFunction(x1,x2):
    w1,w2,b = 2,5,10
    return w1*(20-x1) + w2*x2 + b

def CreateSampleData(m):
    file = Path(file_name)
    if file.exists():
        data = np.load(file)
        X = data["data"]
        Y = data["label"]
    else:
        X = np.zeros((m,2))
        # radius [2,20]
        X[:,0:1] = (np.random.random(1000)*20+2).reshape(1000,1)
        # [40,120] square
        X[:,1:2] = np.random.randint(40,120,(m,1))
        Y = TargetFunction(X[:,0:1], X[:,1:2])
        Noise = np.random.randint(1,100,(m,1)) - 50
        Y = Y + Noise
        np.savez(file_name, data=X, label=Y)
    return X, Y

def NormalizeData(X):
    X_new = np.zeros(X.shape)
    num_feature = X.shape[1]
    X_norm = np.zeros((2,num_feature))
    # 按行归一化,即所有样本的同一特征值分别做归一化
    for i in range(num_feature):
        # get one feature from all examples
        x = X[:,i]
        max_value = np.max(x)
        min_value = np.min(x)
        # min value
        X_norm[0,i] = min_value 
        # range value
        X_norm[1,i] = max_value - min_value 
        x_new = (x - X_norm[0,i])/(X_norm[1,i])
        X_new[:,i] = x_new
    return X_new, X_norm

if __name__ == '__main__':
    X,Y = CreateSampleData(1000)

    print(X[:,0].max())
    print(X[:,0].min())
    print(X[:,0].mean())
    print(X[:,1].max())
    print(X[:,1].min())
    print(X[:,1].mean())
    print(Y.max())
    print(Y.min())
    print(Y.mean())


    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:,0],X[:,1],Y)
    #plt.show()

    XNew, _ = NormalizeData(X)
    YNew, _ = NormalizeData(Y)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(XNew[:,0],XNew[:,1],YNew)
    plt.show()
    