# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from level3_DataNormalization import *

# normalize data by specified range and min_value
def NormalizePredicateData(X_raw, X_norm):
    X_new = np.zeros(X_raw.shape)
    n = X_raw.shape[0]
    for i in range(n):
        x = X_raw[i,:]
        X_new[i,:] = (x-X_norm[0,i])/X_norm[1,i]
    return X_new

# try to give the answer for the price of west(2)，5th ring(5)，93m2
def Inference(W, B, X_norm):
    xt = np.array([2,5,93]).reshape(3,1)
    xt_new = NormalizePredicateData(xt, X_norm)
    z = ForwardCalculationBatch(W, B, xt_new)
    return z

# main
if __name__ == '__main__':
    # hyper parameters
    # SGD, MiniBatch, FullBatch
    method = "SGD"
    # read data
    raw_X, Y = ReadData()
    X,X_norm = NormalizeData(raw_X)
    w, b = train(method,X,Y)
    z = Inference(w,b,X_norm)
    print("z=", z)
