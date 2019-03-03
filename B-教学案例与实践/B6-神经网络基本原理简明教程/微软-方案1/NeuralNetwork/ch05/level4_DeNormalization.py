# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from level3_DataNormalization import *

# get real weights
def DeNormalizeWeights(X_norm, w, b):
    n = w.shape[1]
    W_real = np.zeros((n,))
    for i in range(n):
        W_real[i] = w[0,i] / X_norm[1,i]

    B_real = b
    for i in range(n):
        tmp = w[0,i] * X_norm[0,i] / X_norm[1,i]
        B_real = B_real - tmp
    return W_real, B_real

# 主程序
if __name__ == '__main__':
    # hyper parameters
    # SGD, MiniBatch, FullBatch
    method = "SGD"
    # read data
    raw_X, Y = ReadData()
    X,X_norm = NormalizeData(raw_X)
    w, b = train(method,X,Y)
    W_real, B_real = DeNormalizeWeights(X_norm, w, b)
    print("W_real=", W_real)
    print("B_real=", B_real)

