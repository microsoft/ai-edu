# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
from LossFunction import * 
from Utility import *
from Activations import *

x_data_name = "CurveX.dat"
y_data_name = "CurveY.dat"


def ForwardCalculationBatch(batch_x, dict_weights):
    W1 = dict_weights["W1"]
    B1 = dict_weights["B1"]
    W2 = dict_weights["W2"]
    B2 = dict_weights["B2"]

    Z1 = np.dot(W1, batch_x) + B1
    A1 = CSigmoid().forward(Z1)

    Z2 = np.dot(W2, A1) + B2
    A2 = Z2

    dict_cache ={"A1": A1, "A2": A2, "Output": A2}
    return dict_cache

def BackPropagationBatch(batch_x, batch_y, dict_cache, dict_weights):
    # 批量下降，需要除以样本数量，否则会造成梯度爆炸
    m = batch_x.shape[1]
    # 取出缓存值
    A1 = dict_cache["A1"]
    A2 = dict_cache["A2"]
    W2 = dict_weights["W2"]
    # 第二层的梯度
    dZ2 = A2 - batch_y
    # 第二层的权重和偏移
    dW2 = np.dot(dZ2, A1.T)/m
    dB2 = np.sum(dZ2, axis=1, keepdims=True)/m
    # 第一层的梯度
    dA1 = np.dot(W2.T, dZ2)
    dA1_Z1 = np.multiply(A1, 1 - A1)
    dZ1 = np.multiply(dA1, dA1_Z1)
    # 第一层的权重和偏移
    dW1 = np.dot(dZ1, batch_x.T)/m
    dB1 = np.sum(dZ1, axis=1, keepdims=True)/m
    # 保存到词典中返回
    dict_grads = {"dW1":dW1, "dB1":dB1, "dW2":dW2, "dB2":dB2}
    return dict_grads

def UpdateWeights(dict_weights, dict_grads, learningRate):
    W1 = dict_weights["W1"]
    B1 = dict_weights["B1"]
    W2 = dict_weights["W2"]
    B2 = dict_weights["B2"]

    dW1 = dict_grads["dW1"]
    dB1 = dict_grads["dB1"]
    dW2 = dict_grads["dW2"]
    dB2 = dict_grads["dB2"]

    W1 = W1 - learningRate * dW1
    W2 = W2 - learningRate * dW2
    B1 = B1 - learningRate * dB1
    B2 = B2 - learningRate * dB2

    dict_weights = {"W1": W1,"B1": B1,"W2": W2,"B2": B2}

    return dict_weights


def ShowResult(iteration, neuron, loss, sample_count, dict_weights):
    # draw train data
    plt.scatter(TrainData[0,:],TrainData[1,:], s=1)
    # create and draw visualized validation data
    TX = np.linspace(0,1,100).reshape(1,100)
    dict_cache = ForwardCalculationBatch(TX, dict_weights)
    TY = dict_cache["A2"]
    plt.plot(TX, TY, 'x', c='r')
    plt.title(str.format("neuron={0},example={1},loss={2},iteraion={3}", neuron, sample_count, loss, iteration))
    plt.show()

def train(method, X, Y, dict_params, loss_history):
    num_example = X.shape[1]
    num_feature = X.shape[0]
    num_category = Y.shape[0]
    num_input = dict_params["num_input"]
    num_hidden = dict_params["num_hidden"]
    num_output = dict_params["num_output"]
    batch_size = dict_params["batch_size"]
    max_epoch = dict_params["max_epoch"]
    eta = dict_params["eta"]

    # W(num_category, num_feature), B(num_category, 1)
    dict_weights = InitialParameters(num_input, num_hidden, num_output, 2)

    # calculate loss to decide the stop condition
    loss = 0        # initialize loss (larger than 0)
    error = 0.001    # stop condition

    lossFunc = CLossFunction("MSE")

    # if num_example=200, batch_size=10, then iteration=200/10=20
    max_iteration = (int)(num_example / batch_size)
    for epoch in range(max_epoch):
        for iteration in range(max_iteration):
            # get x and y value for one sample
            batch_x, batch_y = GetBatchSamples(X,Y,batch_size,iteration)
            # get z from x,y
            dict_cache = ForwardCalculationBatch(batch_x, dict_weights)
            # calculate gradient of w and b
            dict_grads = BackPropagationBatch(batch_x, batch_y, dict_cache, dict_weights)
            # update w,b
            dict_weights = UpdateWeights(dict_weights, dict_grads, eta)
        # end for            
        # calculate loss for this batch
        loss = lossFunc.CheckLoss(X, Y, dict_weights)
        print("epoch=%d, loss=%f" %(epoch,loss))
        loss_history.AddLossHistory(loss, dict_weights, epoch, iteration)            
        if math.isnan(loss):
            break
        # end if
        if loss < error:
            break
        # end if
    # end for

# 初始化参数
def InitializeHyperParameters(method, lossFuncType, num_example, num_input, num_hidden, num_output):
    if method=="SGD":
        eta = 0.2
        max_epoch = 50000
        batch_size = 1
    elif method=="MiniBatch":
        eta = 0.2
        max_epoch = 50000
        batch_size = 10
    # end if
    elif method=="FullBatch":
        eta = 0.1
        max_epoch = 50000
        batch_size = num_example
    
    dict_params = {
        "eta": eta, 
        "max_epoch": max_epoch,
        "batch_size": batch_size,
        "num_input": num_input,
        "num_hidden": num_hidden,
        "num_output": num_output,
        "loss_func_type": lossFuncType
    }

    return dict_params

if __name__ == '__main__':

    X,Y = ReadData(x_data_name, y_data_name)
    num_example = X.shape[1]
    n_input, n_hidden, n_output = 1, 64, 1
    # SGD, MiniBatch, FullBatch
    method = "MiniBatch"
    lossFuncType = "MSE"
    dict_params = InitializeHyperParameters(method, lossFuncType, num_example, n_input, n_hidden, n_output)
    loss_history = CLossHistory()
    train(method, X, Y, dict_params, loss_history)

    bookmark = loss_history.GetMinimalLossData()
    print("epoch=%d, iteration=%d, loss=%f" %(bookmark.epoch, bookmark.iteration, bookmark.loss))
    loss_history.ShowLossHistory(method)

    ShowResult(1, n_hidden, 1, 1, bookmark.weights)
    print(bookmark.weights["W1"])
    print(bookmark.weights["B1"])
    print(bookmark.weights["W2"])
    print(bookmark.weights["B2"])