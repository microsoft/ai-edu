# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path

file_name = "../../data/ch05.npz"

# y = w1*x1 + w2*x2 + w3*x3 + b
# W1 = 朝向：1,2,3,4 = N,W,E,S
# W2 = 位置几环：2,3,4,5,6
# W3 = 面积:平米
def TargetFunction(x1,x2,x3):
    w1,w2,w3,b = 2,10,5,10
    return w1*x1 + w2*(10-x2) + w3*x3 + b

def CreateSampleData(m):
    file = Path(file_name)
    if file.exists():
        data = np.load(file)
        X = data["data"]
        Y = data["label"]
    else:
        X = np.zeros((m,3))
        # [1,4] 朝向 direction
        X[:,0:1] = np.random.randint(1,5,(m,1))
        # [2,6] 几环 ring road
        X[:,1:2] = np.random.randint(2,7,(m,1))
        # [40,120] 面积 square
        X[:,2:3] = np.random.randint(40,120,(m,1))
        Y = TargetFunction(X[:,0:1], X[:,1:2], X[:,2:3])
        noise = 20
        Noise = np.random.randint(1,noise,(m,1)) - noise/2
        YData = Y.reshape(1,m)
        np.savez(file_name, data=X, label=Y)
    return X, Y

if __name__ == '__main__':
    CreateSampleData(1000)
