# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
from BaseClassification import *

x_data_name = "X2.dat"
y_data_name = "Y2.dat"

def ToBool(YData):
    num_example = YData.shape[1]
    Y = np.zeros((1, num_example))
    for i in range(num_example):
        if YData[0,i] == 0:     # 第一类的标签设为0
            Y[0,i] = 0
        elif YData[0,i] == 1:   # 第二类的标签设为1
            Y[0,i] = 1
        # end if
    # end for
    return Y

def Sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

# 前向计算
def ForwardCalculationBatch(W, B, batch_X):
    Z = np.dot(W, batch_X) + B
    A = Sigmoid(Z)
    return A

# 计算损失函数值
def CheckLoss(W, B, X, Y):
    m = X.shape[1]
    A = ForwardCalculationBatch(W,B,X)
    
    p1 = 1 - Y
    p2 = np.log(1-A)
    p3 = np.log(A)

    p4 = np.multiply(p1 ,p2)
    p5 = np.multiply(Y, p3)

    LOSS = np.sum(-(p4 + p5))  #binary classification
    loss = LOSS / m
    return loss

def Inference(W,B,X_norm,xt):
    xt_normalized = NormalizePredicateData(xt, X_norm)
    A = ForwardCalculationBatch(W,B,xt_normalized)
    return A, xt_normalized

# 主程序
if __name__ == '__main__':
    # SGD, MiniBatch, FullBatch
    method = "SGD"
    # read data
    XData,YData = ReadData(x_data_name, y_data_name)
    X, X_norm = NormalizeData(XData)
    Y = ToBool(YData)
    W, B = train(method, X, Y, ForwardCalculationBatch, CheckLoss)
    print("W=",W)
    print("B=",B)
    xt = np.array([5,1,6,9,5,5]).reshape(2,3,order='F')
    result, xt_norm = Inference(W,B,X_norm,xt)
    print("result=", result)
    print(np.around(result))




