# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# 用最小二乘法得到解析解LSE

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from HelperClass.DataReader_1_0 import *

file_name = "../../data/ch04.npz"

# 根据公式15
def method1(X,Y,m):
    x_mean = X.mean()
    p = sum(Y*(X-x_mean))
    q = sum(X*X) - sum(X)*sum(X)/m
    w = p/q
    return w

# 根据公式16
def method2(X,Y,m):
    x_mean = X.mean()
    y_mean = Y.mean()
    p = sum(X*(Y-y_mean))
    q = sum(X*X) - x_mean*sum(X)
    w = p/q
    return w

# 根据公式13
def method3(X,Y,m):
    p = m*sum(X*Y) - sum(X)*sum(Y)
    q = m*sum(X*X) - sum(X)*sum(X)
    w = p/q
    return w

# 根据公式14
def calculate_b_1(X,Y,w,m):
    b = sum(Y-w*X)/m
    return b

# 根据公式9
def calculate_b_2(X,Y,w):
    b = Y.mean() - w * X.mean()
    return b

if __name__ == '__main__':

    reader = DataReader_1_0(file_name)
    reader.ReadData()
    X,Y = reader.GetWholeTrainSamples()
    m = X.shape[0]
    w1 = method1(X,Y,m)
    b1 = calculate_b_1(X,Y,w1,m)

    w2 = method2(X,Y,m)
    b2 = calculate_b_2(X,Y,w2)

    w3 = method3(X,Y,m)
    b3 = calculate_b_1(X,Y,w3,m)

    print("w1=%f, b1=%f" % (w1,b1))
    print("w2=%f, b2=%f" % (w2,b2))
    print("w3=%f, b3=%f" % (w3,b3))

