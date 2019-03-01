# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from level2_NeuralNetwork import *

# normalize data by extracting range from source data
def NormalizeData(X):
    X_new = np.zeros(X.shape)
    n = X.shape[0]
    x_range = np.zeros((1,n))
    x_min = np.zeros((1,n))
    for i in range(n):
        x = X[i,:]
        max_value = np.max(x)
        min_value = np.min(x)
        x_min[0,i] = min_value
        x_range[0,i] = max_value - min_value
        x_new = (x - x_min[0,i])/(x_range[0,i])
        X_new[i,:] = x_new
    return X_new, x_range, x_min


# normalize data by specified range and min_value
def NormalizeDataByRange(X, x_range, x_min):
    X_new = np.zeros(X.shape)
    n = X.shape[0]
    for i in range(n):
        x = X[i,:]
        x_new = (x-x_min[0,i])/x_range[0,i]
        X_new[i,:] = x_new
    return X_new

# get real weights
def DeNormalizeWeights(X_range, XData, n):
    W_real = np.zeros((1,n))
    for i in range(n):
        W_real[0,i] = W[0,i] / X_range[0,i]
    print("W_real=", W_real)

    B_real = 0
    num_sample = XData.shape[1]
    for i in range(num_sample):
        xm = XData[0:n,i].reshape(n,1)
        zm = ForwardCalculationBatch(xm, W_real, 0)
        ym = Y[0,i].reshape(1,1)
        B_real = B_real + (ym - zm)
    B_real = B_real / num_sample
    print("B_real=", B_real)
    return W_real, B_real

# 主程序
if __name__ == '__main__':
    # hyper parameters
    # SGD, MiniBatch, FullBatch
    method = "SGD"

    eta, max_epoch,batch_size = InitializeHyperParameters(method)
    
    # W size is 3x1, B is 1x1
    W, B = InitialWeights(3,1,2)
    # calculate loss to decide the stop condition
    error = 1e-5
    loss = 10 # set to any number bigger than error value
    dict_loss = {}
    # read data
    raw_X, Y = ReadData()
    X,X_range,X_min = NormalizeData(raw_X)
    # count of samples
    num_example = X.shape[1]
    num_feature = X.shape[0]

    # if num_example=200, batch_size=10, then iteration=200/10=20
    max_iteration = (int)(num_example / batch_size)
    for epoch in range(max_epoch):
        print("epoch=%d" %epoch)
        for iteration in range(max_iteration):
            # get x and y value for one sample
            batch_x, batch_y = GetBatchSamples(X,Y,batch_size,iteration)
            # get z from x,y
            batch_z = ForwardCalculationBatch(W, B, batch_x)
            # calculate gradient of w and b
            dW, dB = BackPropagationBatch(batch_x, batch_y, batch_z)
            # update w,b
            W, B = UpdateWeights(W, B, dW, dB, eta)
            
            # calculate loss for this batch
            loss = CheckLoss(W,B,X,Y)
            print(epoch,iteration,loss,W,B)
            dict_loss[loss] = CData(loss, W, B, epoch, iteration)            
        if loss < error:
            break

    ShowLossHistory(dict_loss, method)
    w,b,cdata = GetMinimalLossData(dict_loss)
    print("w=", cdata.w)
    print("b=", cdata.b)
    print("epoch=%d, iteration=%d, loss=%f" %(cdata.epoch, cdata.iteration, cdata.loss))




#    W_real, B_real = DeNormalizeWeights(X_range, raw_X, num_feature)
#    print(W_real, B_real)

