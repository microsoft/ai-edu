# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_name = "../../data/ch05.npz"

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
    plt.show()
    