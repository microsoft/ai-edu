# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

def Sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

# 前向计算
def ForwardCalculationBatch(W, B, batch_X):
    Z = np.dot(W, batch_X) + B
    A = Sigmoid(Z)
    return A

# 反向计算
def BackPropagationBatch(batch_X, batch_Y, A):
    m = batch_X.shape[1]
    dZ = A - batch_Y
    # dZ列相加，即一行内的所有元素相加
    dB = dZ.sum(axis=1, keepdims=True)/m
    dW = np.dot(dZ, batch_X.T)/m
    return dW, dB

# 更新权重参数
def UpdateWeights(W, B, dW, dB, eta):
    W = W - eta * dW
    B = B - eta * dB
    return W, B

# 计算损失函数值
def CheckLoss(W, B, X, Y):
    m = X.shape[1]
    A = ForwardCalculationBatch(W,B,X)
    
    p4 = np.multiply(1-Y ,np.log(1-A))
    p5 = np.multiply(Y, np.log(A))

    LOSS = np.sum(-(p4 + p5))  #binary classification
    loss = LOSS / m
    return loss

# 初始化权重值
def InitialWeights(num_input, num_output, method):
    if method == "zero":
        # zero
        W = np.zeros((num_output, num_input))
    elif method == "norm":
        # normalize
        W = np.random.normal(size=(num_output, num_input))
    elif method == "xavier":
        # xavier
        W=np.random.uniform(
            -np.sqrt(6/(num_input+num_output)),
            np.sqrt(6/(num_input+num_output)),
            size=(num_output,num_input))

    B = np.zeros((num_output, 1))
    return W,B


def train(X, Y, ForwardCalculationBatch, CheckLoss):
    num_example = X.shape[1]
    num_feature = X.shape[0]
    num_category = Y.shape[0]
    # hyper parameters
    eta = 0.5
    max_epoch = 10000
    # W(num_category, num_feature), B(num_category, 1)
    W, B = InitialWeights(num_feature, num_category, "zero")
    # calculate loss to decide the stop condition
    loss = 5        # initialize loss (larger than 0)
    error = 2e-3    # stop condition

    # if num_example=200, batch_size=10, then iteration=200/10=20
    for epoch in range(max_epoch):
        print("epoch=%d" %epoch)
        for i in range(num_example):
            # get x and y value for one sample
            x = X[:,i].reshape(num_feature,1)
            y = Y[:,i].reshape(1,1)
            # get z from x,y
            batch_a = ForwardCalculationBatch(W, B, x)
            # calculate gradient of w and b
            dW, dB = BackPropagationBatch(x, y, batch_a)
            # update w,b
            W, B = UpdateWeights(W, B, dW, dB, eta)
            
            # calculate loss for this batch
            loss = CheckLoss(W,B,X,Y)
            if i % 10 == 0:
                print(epoch,i,loss,W,B)
            # end if
        # end for
        if loss < error:
            break
    # end for

    return W,B

def ShowResult(W,B,X,Y):
    w = -W[0,0]/W[0,1]
    b = -B[0,0]/W[0,1]
    x = np.array([0,1])
    y = w * x + b
    plt.plot(x,y)
   
    for i in range(X.shape[1]):
        if Y[0,i] == 0:
            plt.scatter(X[0,i],X[1,i],marker="o",c='b',s=64)
        else:
            plt.scatter(X[0,i],X[1,i],marker="^",c='r',s=64)
    plt.axis([-0.1,1.1,-0.1,1.1])
    plt.show()
