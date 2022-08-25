# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


file_name = "../../data/ch06.npz"

def TargetFunction(x1,x2):
    y = 2*x1-0.5
    if x2 > y:
        return 1
    else:
        return 0

def CreateSampleData(m):
    file = Path(file_name)
    if file.exists():
        data = np.load(file)
        X = data["data"]
        Y = data["label"]
    else:
        X = np.random.random((200,2))
        Y = np.zeros((200,1))
        for i in range(200):
            y = TargetFunction(X[i,0], X[i,1])
            Y[i,0] = y
        np.savez(file_name, data=X, label=Y)
    return X, Y

if __name__ == '__main__':
    X,Y = CreateSampleData(200)
    fig = plt.figure(figsize=(6.5,6.5))
    for i in range(200):
        if Y[i,0] == 1:
            plt.scatter(X[i,0], X[i,1], marker='x', c='g')
        else:
            plt.scatter(X[i,0], X[i,1], marker='o', c='r')
    plt.show()