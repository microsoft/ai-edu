# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
from BaseClassification import *

x_data_name = "Pollution2CategoryX.dat"
y_data_name = "Pollution2CategoryY.dat"

def ToBool(YData):
    num_example = YData.shape[1]
    Y = np.zeros((1, num_example))
    for i in range(num_example):
        if YData[0,i] == 1:     # 第一类的标签设为0
            Y[0,i] = 0
        elif YData[0,i] == 2:   # 第二类的标签设为1
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
    p1 = (1-Y) * np.log(1-A)   #binary classification
    p2 =  Y * np.log(A)          # binary/multiple classification
    LOSS = np.sum(-(p1 + p2))  #binary classification
    loss = LOSS / m
    return loss

def ShowResult(X,Y,W,B):
    for i in range(X.shape[1]):
        if Y[0,i] == 1:
            plt.scatter(X[0,i], X[1,i], c='b')
        elif Y[0,i] == 2:
            plt.scatter(X[0,i], X[1,i], c='g')
        # end if
    # end for

    b12 = -B[0,0]/W[0,1]
    w12 = -W[0,0]/W[0,1]

    x = np.linspace(0,1,10)
    y = w12 * x + b12
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
    ShowData(XData, YData)
    num_category = 2
    Y = ToBool(YData)
    W, B = train(method, X, Y, ForwardCalculationBatch, CheckLoss)
    print(W, B)
    ShowResult(X,YData,W,B)



