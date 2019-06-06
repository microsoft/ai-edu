# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

file_name = "../../data/ch04.npz"

def TargetFunction(X):
    noise = np.random.normal(0,0.2,X.shape)
    W = 2
    B = 3
    Y = np.dot(X, W) + B + noise
    return Y

def CreateSampleData(m):
    file = Path(file_name)
    if file.exists():
        data = np.load(file)
        X = data["data"]
        Y = data["label"]
    else:
        X = np.random.random((m,1))
        Y = TargetFunction(X)
        np.savez(file_name, data=X, label=Y)
    return X, Y

if __name__ == '__main__':
    X,Y = CreateSampleData(100)
    plt.scatter(X,Y,s=10)
    plt.title("Air Conditioner Power")
    plt.xlabel("Number of Servers(K)")
    plt.ylabel("Power of Air Conditioner(KW)")
    plt.show()



