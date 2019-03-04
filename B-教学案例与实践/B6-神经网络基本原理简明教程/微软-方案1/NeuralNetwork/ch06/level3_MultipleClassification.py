# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
from BaseClassification import *

x_data_name = "Pollution3CategoryX.dat"
y_data_name = "Pollution3CategoryY.dat"

def ToOneHot(YData, num_category):
    num_example = YData.shape[1]
    Y = np.zeros((num_category, num_example))
    for i in range(num_example):
        if YData[0,i] == 1:
            Y[0,i] = 1
        elif YData[0,i] == 2:
            Y[1,i] = 1
        elif YData[0,i] == 3:
            Y[2,i] = 1
        # end if
    # end for
    return Y

def Softmax(Z):
    shift_z = Z - np.max(Z, axis=0)
    exp_z = np.exp(shift_z)
    A = exp_z / np.sum(exp_z, axis=0)
    return A

# 前向计算
def ForwardCalculationBatch(W, B, batch_X):
    Z = np.dot(W, batch_X) + B
    A = Softmax(Z)
    return A

# 计算损失函数值
def CheckLoss(W, B, X, Y):
    m = X.shape[1]
    A = ForwardCalculationBatch(W,B,X)
    p2 =  Y * np.log(A)          # multiple classification
    LOSS = np.sum(-p2)          # multiple classification
    loss = LOSS / m
    return loss

def ShowResult(X,Y,W,B):
    for i in range(X.shape[1]):
        if Y[0,i] == 1:
            plt.scatter(X[0,i], X[1,i], c='b')
        elif Y[0,i] == 2:
            plt.scatter(X[0,i], X[1,i], c='g')
        elif Y[0,i] == 3:
            plt.scatter(X[0,i], X[1,i], c='r')
        # end if
    # end for

    b13 = (B[0,0] - B[2,0])/(W[2,1] - W[0,1])
    w13 = (W[0,0] - W[2,0])/(W[2,1] - W[0,1])

    b23 = (B[2,0] - B[1,0])/(W[1,1] - W[2,1])
    w23 = (W[2,0] - W[1,0])/(W[1,1] - W[2,1])

    x = np.linspace(0,1,10)
    y = w13 * x + b13
    plt.plot(x,y)

    x = np.linspace(0,1,10)
    y = w23 * x + b23
    plt.plot(x,y)

#    title = str.format("eta:{0}, iteration:{1}, eps:{2}", eta, iteration, eps)
#    plt.title(title)
    
    plt.xlabel("Temperature")
    plt.ylabel("Humidity")
    plt.show()

# 主程序
if __name__ == '__main__':
    # SGD, MiniBatch, FullBatch
    method = "SGD"
    # read data
    XData,YData = ReadData(x_data_name, y_data_name)
    X, X_norm = NormalizeData(XData)
    ShowData(X, YData)
    num_category = 3
    Y = ToOneHot(YData, num_category)
    W, B = train(method, X, Y, ForwardCalculationBatch, CheckLoss)
    print(W, B)
    ShowResult(X,YData,W,B)



