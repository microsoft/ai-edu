# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
from LossHistory import * 

train_data_name = "CurveFittingTrainData.npy"


def ReadData():
    Trainfile = Path(train_data_name)
    if Trainfile.exists():
        TrainData = np.load(Trainfile)
        return TrainData
    
    return None

def ForwardCalculationBatch(batch_x, dict_weights):
    W1 = dict_weights["W1"]
    B1 = dict_weights["B1"]
    W2 = dict_weights["W2"]
    B2 = dict_weights["B2"]

    Z1 = np.dot(W1, batch_x) + B1
    A1 = Sigmoid(Z1)

    Z2 = np.dot(W2, A1) + B2
    A2 = Z2

    dict_cache ={"A1": A1, "A2": A2}
    return dict_cache

def BackPropagationBatch(batch_x, batch_y, dict_cache, dict_weights):

    m = batch_x.shape[1]

    A1 = dict_cache["A1"]
    A2 = dict_cache["A2"]
    W2 = dict_weights["W2"]

    dLoss_Z2 = A2 - batch_y
    dW2 = np.dot(dLoss_Z2, A1.T)/m
    dB2 = np.sum(dLoss_Z2, axis=1, keepdims=True)/m

    dLoss_A1 = np.dot(W2.T, dLoss_Z2)
    dA1_Z1 = np.multiply(A1, 1 - A1)
    
    dLoss_Z1 = np.multiply(dLoss_A1, dA1_Z1)
    dW1 = np.dot(dLoss_Z1, batch_x.T)/m
    dB1 = np.sum(dLoss_Z1, axis=1, keepdims=True)/m

    dict_grads = {"dW1":dW1, "dB1":dB1, "dW2":dW2, "dB2":dB2}
    return dict_grads

def CheckLoss(X, Y, dict_weights):
    m = X.shape[1]
    dict_cache = ForwardCalculationBatch(X, dict_weights)
    A2 = dict_cache["A2"]
    p1 = A2 - Y
    LOSS = np.multiply(p1, p1)
    loss = LOSS.sum()/m/2
    return loss

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

def Sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a

def InitialParameters(num_input, num_hidden, num_output, flag):
    if flag == 0:
        # zero
        W1 = np.zeros((num_hidden, num_input))
        W2 = np.zeros((num_output, num_hidden))
    elif flag == 1:
        # normalize
        W1 = np.random.normal(size=(num_hidden, num_input))
        W2 = np.random.normal(size=(num_output, num_hidden))
    elif flag == 2:
        #
        W1=np.random.uniform(-np.sqrt(6)/np.sqrt(num_input+num_hidden),np.sqrt(6)/np.sqrt(num_hidden+num_input),size=(num_hidden,num_input))
        W2=np.random.uniform(-np.sqrt(6)/np.sqrt(num_output+num_hidden),np.sqrt(6)/np.sqrt(num_output+num_hidden),size=(num_output,num_hidden))

    B1 = np.zeros((num_hidden, 1))
    B2 = np.zeros((num_output, 1))
    dict_Param = {"W1": W1, "B1": B1, "W2": W2, "B2": B2}
    return dict_Param

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

# 获得批样本数据
def GetBatchSamples(X,Y,batch_size,iteration):
    num_feature = X.shape[0]
    start = iteration * batch_size
    end = start + batch_size
    batch_X = X[0:num_feature, start:end].reshape(num_feature, batch_size)
    batch_Y = Y[:, start:end].reshape(-1, batch_size)
    return batch_X, batch_Y

# 初始化参数
def InitializeHyperParameters(method, num_example):
    if method=="SGD":
        eta = 0.2
        max_epoch = 50000
        batch_size = 1
    elif method=="MiniBatch":
        eta = 0.2
        max_epoch = 50000
        batch_size = 10
    # end if
#    elif method=="FullBatch":
#        eta = 0.1
#        max_epoch = 10000
#        batch_size = num_example
    return eta, max_epoch, batch_size


def train(method, X, Y, num_input, num_hidden, num_output, loss_history):
    num_example = X.shape[1]
    num_feature = X.shape[0]
    num_category = Y.shape[0]
    # hyper parameters
    eta, max_epoch,batch_size = InitializeHyperParameters(method,num_example)
    # W(num_category, num_feature), B(num_category, 1)
    dict_weights = InitialParameters(num_input, num_hidden, num_output, 2)

    # calculate loss to decide the stop condition
    loss = 0        # initialize loss (larger than 0)
    error = 0.001    # stop condition

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
        loss = CheckLoss(X, Y, dict_weights)
        print("epoch=%d, loss=%f" %(epoch,loss))
        loss_history.AddLossHistory(loss, dict_weights, epoch, iteration)            
        if math.isnan(loss):
            break
        # end if
        if loss < error:
            break
        # end if
    # end for

if __name__ == '__main__':
    loss_history = CLossHistory()
    # SGD, MiniBatch, FullBatch
    method = "MiniBatch"
    n_input, n_hidden, n_output = 1, 8, 1

    TrainData = ReadData()
    num_samples = TrainData.shape[1]
    X = TrainData[0,:].reshape(1, num_samples)
    Y = TrainData[1,:].reshape(1, num_samples)
    
    train(method, X, Y, n_input, n_hidden, n_output, loss_history)

    bookmark = loss_history.GetMinimalLossData()
    print("epoch=%d, iteration=%d, loss=%f" %(bookmark.epoch, bookmark.iteration, bookmark.loss))
    loss_history.ShowLossHistory(method)

    ShowResult(1, n_hidden, 1, 1, bookmark.weights)
    print(bookmark.weights["W1"])
    print(bookmark.weights["B1"])
    print(bookmark.weights["W2"])
    print(bookmark.weights["B2"])