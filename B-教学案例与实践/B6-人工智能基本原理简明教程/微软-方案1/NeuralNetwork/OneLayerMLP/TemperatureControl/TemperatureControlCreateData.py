# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

x_data_name = "TemperatureControlXData.npy"
y_data_name = "TemperatureControlYData.npy"

def TargetFunction(X):
    noise = np.random.normal(0,0.1,X.shape)
    W = 2
    B = 3
    Y = W * X + B + noise

def CreateSampleData(m):
    Xfile = Path(x_data_name)
    Yfile = Path(y_data_name)
    if Xfile.exists() & Yfile.exists():
        X = np.load(Xfile)
        Y = np.load(Yfile)
    else:
        X = np.random.random(m)
        Y = TargetFunction(X)
        np.save(x_data_name, X)
        np.save(y_data_name, Y)
    return X, Y

X,Y = CreateSampleData(200)
plt.scatter(X,Y,c="b")
plt.title("Air Conditioner Power")
plt.xlabel("Number of Servers(K)")
plt.ylabel("Power of Air Conditioner(KW)")
plt.show()
print(X)
print(Y)


