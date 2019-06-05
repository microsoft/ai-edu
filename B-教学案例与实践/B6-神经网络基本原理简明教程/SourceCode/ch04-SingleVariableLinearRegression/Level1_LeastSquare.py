# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# 用最小二乘法得到解析解LSE

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SimpleDataReader import *

file_name = "../../data/ch04.npz"

# 根据公式8
def method1(X,Y,m):
    x_mean = np.mean(X)
    p = sum(Y*(X-x_mean))
    q = sum(X*X) - sum(X)*sum(X)/m
    w = p/q
    return w

# 根据公式9
def method2(X,Y,m):
    x_mean = sum(X)/m
    y_mean = sum(Y)/m
    p = sum(X*(Y-y_mean))
    q = sum(X*X) - x_mean*sum(X)
    w = p/q
    return w

# 根据公式6
def method3(X,Y,m):
    m = X.shape[0]
    p = m*sum(X*Y) - sum(X)*sum(Y)
    q = m*sum(X*X) - sum(X)*sum(X)
    w = p/q
    return w

# 根据公式7
def calculate_b(X,Y,w,m):
    b = sum(Y-w*X)/m
    return b

if __name__ == '__main__':

    reader = SimpleDataReader(file_name)
    reader.ReadData()
    X,Y = reader.GetWholeTrainSamples()
    m = X.shape[0]
    w1 = method1(X,Y,m)
    b1 = calculate_b(X,Y,w1,m)

    w2 = method2(X,Y,m)
    b2 = calculate_b(X,Y,w2,m)

    w3 = method3(X,Y,m)
    b3 = calculate_b(X,Y,w3,m)

    print("w1=%f, b1=%f" % (w1,b1))
    print("w2=%f, b2=%f" % (w2,b2))
    print("w3=%f, b3=%f" % (w3,b3))

