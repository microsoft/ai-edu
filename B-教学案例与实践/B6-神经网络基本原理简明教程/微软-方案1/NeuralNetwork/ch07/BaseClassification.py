# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

# 帮助类，用于记录损失函数值极其对应的权重/迭代次数
class CData(object):
    def __init__(self, loss, w, b, epoch, iteration):
        self.loss = loss
        self.w = w
        self.b = b
        self.epoch = epoch
        self.iteration = iteration

# binary classification
def ReadData(x_data_name, y_data_name):
    Xfile = Path(x_data_name)
    Yfile = Path(y_data_name)
    if Xfile.exists() & Yfile.exists():
        XRawData = np.load(Xfile)
        YRawData = np.load(Yfile)
        return XRawData,YRawData
    # end if
    return None,None

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
    # end for
    return X_new, X_norm

# normalize data by specified range and min_value
def NormalizePredicateData(X_raw, X_norm):
    X_new = np.zeros(X_raw.shape)
    num_feature = X_raw.shape[0]
    for i in range(num_feature):
        x = X_raw[i,:]
        X_new[i,:] = (x-X_norm[0,i])/X_norm[1,i]
    return X_new

# 反向计算
# X:input example, Y:lable example, Z:predicated value
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

# 获得批样本数据
def GetBatchSamples(X,Y,batch_size,iteration):
    num_feature = X.shape[0]
    start = iteration * batch_size
    end = start + batch_size
    batch_X = X[0:num_feature, start:end].reshape(num_feature, batch_size)
    batch_Y = Y[:, start:end].reshape(-1, batch_size)
    return batch_X, batch_Y

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

# 初始化参数
def InitializeHyperParameters(method, num_example):
    if method=="SGD":
        eta = 0.1
        max_epoch = 100
        batch_size = 1
    elif method=="MiniBatch":
        eta = 0.1
        max_epoch = 50
        batch_size = 10
    elif method=="FullBatch":
        eta = 0.5
        max_epoch = 100
        batch_size = num_example
    return eta, max_epoch, batch_size

# 从历史记录中获得最小损失值得训练权重值
def GetMinimalLossData(dict_loss):
    key = sorted(dict_loss.keys())[0]
    w = dict_loss[key].w
    b = dict_loss[key].b
    return w,b,dict_loss[key]

# 图形显示损失函数值历史记录
def ShowLossHistory(dict_loss, method):
    loss = []
    for key in dict_loss:
        loss.append(key)

    #plt.plot(loss)
    plt.plot(loss)
    plt.title(method)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.show()

def ShowData(X,Y):
    for i in range(X.shape[1]):
        if Y[0,i] == 0:
            plt.plot(X[0,i], X[1,i], '.', c='r')
        elif Y[0,i] == 1:
            plt.plot(X[0,i], X[1,i], 'x', c='g')
        elif Y[0,i] == 2:
            plt.plot(X[0,i], X[1,i], '^', c='b')
        # end if
    # end for
    plt.show()


def train(method, X, Y, ForwardCalculationBatch, CheckLoss):
    num_example = X.shape[1]
    num_feature = X.shape[0]
    num_category = Y.shape[0]
    # hyper parameters
    eta, max_epoch,batch_size = InitializeHyperParameters(method,num_example)
    # W(num_category, num_feature), B(num_category, 1)
    W, B = InitialWeights(num_feature, num_category, "zero")
    # calculate loss to decide the stop condition
    loss = 5        # initialize loss (larger than 0)
    error = 1e-3    # stop condition
    dict_loss = {}  # loss history

    # if num_example=200, batch_size=10, then iteration=200/10=20
    max_iteration = (int)(num_example / batch_size)
    for epoch in range(max_epoch):
        print("epoch=%d" %epoch)
        for iteration in range(max_iteration):
            # get x and y value for one sample
            batch_x, batch_y = GetBatchSamples(X,Y,batch_size,iteration)
            # get z from x,y
            batch_a = ForwardCalculationBatch(W, B, batch_x)
            # calculate gradient of w and b
            dW, dB = BackPropagationBatch(batch_x, batch_y, batch_a)
            # update w,b
            W, B = UpdateWeights(W, B, dW, dB, eta)
            
            # calculate loss for this batch
            loss = CheckLoss(W,B,X,Y)
            if method == "SGD":
                if iteration % 10 == 0:
                    print(epoch,iteration,loss,W,B)
                # end if
            else:
                print(epoch,iteration,loss,W,B)
            # end if
            dict_loss[loss] = CData(loss, W, B, epoch, iteration)            
        # end for
        if math.isnan(loss):
            break
        if loss < error:
            break
    # end for

    ShowLossHistory(dict_loss, method)
    w,b,cdata = GetMinimalLossData(dict_loss)
    print(cdata.w, cdata.b)
    print("epoch=%d, iteration=%d, loss=%f" %(cdata.epoch, cdata.iteration, cdata.loss))
    return w,b
