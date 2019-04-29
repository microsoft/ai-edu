# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from level2_NeuralNetwork import *

# normalize data by extracting range from source data
# return: X_new: normalized data with same shape
# return: X_norm: 2xn
#               [[min1, min2, min3...]
#                [range1, range2, range3...]]
def NormalizeData(X):
    X_new = np.zeros(X.shape)
    num_feature = X.shape[0]
    X_norm = np.zeros((2,num_feature))
    # 按行归一化,即所有样本的同一特征值分别做归一化
    for i in range(num_feature):
        # get one feature from all examples
        x = X[i,:]
        max_value = np.max(x)
        min_value = np.min(x)
        # min value
        X_norm[0,i] = min_value 
        # range value
        X_norm[1,i] = max_value - min_value 
        x_new = (x - X_norm[0,i])/(X_norm[1,i])
        X_new[i,:] = x_new
    return X_new, X_norm

# 主程序
if __name__ == '__main__':
    # hyper parameters
    # SGD, MiniBatch, FullBatch
    method = "SGD"
    # read data
    raw_X, Y = ReadData()
    X,X_norm = NormalizeData(raw_X)
    w, b = train(method,X,Y)
