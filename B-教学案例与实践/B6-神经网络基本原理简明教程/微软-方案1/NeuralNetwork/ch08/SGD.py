# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

train_data_name = "CurveFittingTrainData.npy"

def ReadData():
    Trainfile = Path(train_data_name)
    if Trainfile.exists():
        TrainData = np.load(Trainfile)
        return TrainData
    
    return None

def ForwardCalculation(X, dictWeights):
    W1 = dictWeights["W1"]
    B1 = dictWeights["B1"]
    W2 = dictWeights["W2"]
    B2 = dictWeights["B2"]

    Z1 = np.dot(W1,X) + B1
    A1 = sigmoid(Z1)

    Z2 = np.dot(W2,A1) + B2
    A2 = Z2

    dictCache ={"A1": A1, "A2": A2}
    return A2, dictCache

def BackPropagation(X, Y, dictCache, dictWeights):
    A1 = dictCache["A1"]
    A2 = dictCache["A2"]
    W2 = dictWeights["W2"]

    dLoss_Z2 = (A2 - Y)
    dW2 = np.dot(dLoss_Z2, A1.T)
    dB2 = np.sum(dLoss_Z2, axis=1, keepdims=True)

    dLoss_A1 = np.dot(W2.T, dLoss_Z2)
    dA1_Z1 = A1 * (1 - A1)
    
    dLoss_Z1 = dLoss_A1 * dA1_Z1
    dW1 = np.dot(dLoss_Z1, X.T)
    dB1 = np.sum(dLoss_Z1, axis=1, keepdims=True)

    dictGrads = {"dW1":dW1, "dB1":dB1, "dW2":dW2, "dB2":dB2}
    return dictGrads

def LossCalculation(X, Y, dictWeights, prev_loss, count):
    A2, dict_Cache = ForwardCalculation(X, dictWeights)
    LOSS = (A2 - Y)**2
    loss = LOSS.sum()/count/2
    diff_loss = abs(loss - prev_loss)
    return loss, diff_loss

def UpdateWeights(dictWeights, dictGrads, learningRate):
    W1 = dictWeights["W1"]
    B1 = dictWeights["B1"]
    W2 = dictWeights["W2"]
    B2 = dictWeights["B2"]

    dW1 = dictGrads["dW1"]
    dB1 = dictGrads["dB1"]
    dW2 = dictGrads["dW2"]
    dB2 = dictGrads["dB2"]

    W1 = W1 - learningRate * dW1
    W2 = W2 - learningRate * dW2
    B1 = B1 - learningRate * dB1
    B2 = B2 - learningRate * dB2

    dictWeights = {"W1": W1,"B1": B1,"W2": W2,"B2": B2}

    return dictWeights

def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

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

def ShowResult(iteration, neuron, loss, sample_count):
    # draw train data
    plt.scatter(TrainData[0,:],TrainData[1,:], s=1)
    # create and draw visualized validation data
    TX = np.linspace(0,1,100).reshape(1,100)
    TY, cache = ForwardCalculation(TX, dictWeights)
    plt.scatter(TX, TY, c='r', s=2)
    plt.title(str.format("neuron={0},example={1},loss={2},iteraion={3}", neuron, sample_count, loss, iteration))
    plt.show()

TrainData = ReadData()
num_samples = TrainData.shape[1]
X = TrainData[0,:].reshape(1, num_samples)
Y = TrainData[1,:].reshape(1, num_samples)

n_input, n_hidden, n_output = 1, 128, 1
learning_rate = 0.1
eps = 1e-10
dictWeights = InitialParameters(n_input, n_hidden, n_output, 1)
max_iteration = 10000
min_loss = 0.002
loss, prev_loss, diff_loss = 0, 0, 10
loop = num_samples
for iteration in range(max_iteration):
    for i in range(loop):
        x = X[0,i]
        y = Y[0,i]
        A2, dictCache = ForwardCalculation(x, dictWeights)
        dictGrads = BackPropagation(x, y, dictCache, dictWeights)
        dictWeights = UpdateWeights(dictWeights, dictGrads, learning_rate)
   
    loss, diff_loss = LossCalculation(X, Y, dictWeights, prev_loss, num_samples)
    print(iteration,loss,diff_loss)
    if diff_loss < eps:
        break
    if loss < min_loss:
        break
    prev_loss = loss

print(loss, diff_loss)
ShowResult(iteration+1, n_hidden, min_loss, loop)

