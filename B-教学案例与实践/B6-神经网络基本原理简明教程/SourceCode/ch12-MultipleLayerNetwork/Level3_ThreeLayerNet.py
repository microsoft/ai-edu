# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import struct
import matplotlib.pyplot as plt

from Level1_Base import *

def Tanh(z):
    a = 2.0 / (1.0 + np.exp(-2*z)) - 1.0
    return a

def forward3(X, dict_Param):
    W1 = dict_Param["W1"]
    B1 = dict_Param["B1"]
    W2 = dict_Param["W2"]
    B2 = dict_Param["B2"]
    W3 = dict_Param["W3"]
    B3 = dict_Param["B3"]
    
    Z1 = np.dot(W1,X) + B1
    A1 = Sigmoid(Z1)

    Z2 = np.dot(W2,A1) + B2
    A2 = Tanh(Z2)

    Z3 = np.dot(W3,A2) + B3
    A3 = Softmax(Z3)
    
    dict_Cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3, "Output": A3}
    return dict_Cache

def backward3(dict_Param,cache,X,Y):
    W1 = dict_Param["W1"]
    W2 = dict_Param["W2"]
    W3 = dict_Param["W3"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]

    m = X.shape[1]

    dZ3= A3 - Y
    dW3 = np.dot(dZ3, A2.T)/m
    dB3 = np.sum(dZ3, axis=1, keepdims=True)/m

    # dZ2 = W3T * dZ3 * dA3
    dZ2 = np.dot(W3.T, dZ3) * (1-A2*A2) # tanh
    dW2 = np.dot(dZ2, A1.T)/m
    dB2 = np.sum(dZ2, axis=1, keepdims=True)/m

    # dZ1 = W2T * dZ2 * dA2
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1-A1)   #sigmoid
    dW1 = np.dot(dZ1, X.T)/m
    dB1 = np.sum(dZ1, axis=1, keepdims=True)/m

    dict_Grads = {"dW1": dW1, "dB1": dB1, "dW2": dW2, "dB2": dB2, "dW3": dW3, "dB3": dB3}
    return dict_Grads

def update3(dict_Param, dict_Grads, learning_rate):
    W1 = dict_Param["W1"]
    B1 = dict_Param["B1"]
    W2 = dict_Param["W2"]
    B2 = dict_Param["B2"]
    W3 = dict_Param["W3"]
    B3 = dict_Param["B3"]

    dW1 = dict_Grads["dW1"]
    dB1 = dict_Grads["dB1"]
    dW2 = dict_Grads["dW2"]
    dB2 = dict_Grads["dB2"]
    dW3 = dict_Grads["dW3"]
    dB3 = dict_Grads["dB3"]

    W1 = W1 - learning_rate * dW1
    B1 = B1 - learning_rate * dB1
    W2 = W2 - learning_rate * dW2
    B2 = B2 - learning_rate * dB2
    W3 = W3 - learning_rate * dW3
    B3 = B3 - learning_rate * dB3

    dict_Param = {"W1": W1, "B1": B1, "W2": W2, "B2": B2, "W3": W3, "B3": B3}
    return dict_Param

def InitialParameters3(num_input, num_hidden1, num_hidden2, num_output, flag):
    if flag == 0:
        # zero
        W1 = np.zeros((num_hidden1, num_input))
        W2 = np.zeros((num_hidden2, num_hidden1))
        W3 = np.zeros((num_output, num_hidden2))
    elif flag == 1:
        # normalize
        W1 = np.random.normal(size=(num_hidden1, num_input))
        W2 = np.random.normal(size=(num_hidden2, num_hidden1))
        W3 = np.random.normal(size=(num_output, num_hidden2))
    elif flag == 2:
        #
        t1 = np.sqrt(6/(num_input + num_hidden1))
        W1 = np.random.uniform(-t1, t1, size=(num_hidden1, num_input))

        t2 = np.sqrt(6/(num_hidden2 + num_hidden1))
        W2 = np.random.uniform(-t2, t2, size=(num_hidden2, num_hidden1))

        t3 = np.sqrt(6/(num_output + num_hidden2))
        W3 = np.random.uniform(-t3, t3, size=(num_output, num_hidden2))

    B1 = np.zeros((num_hidden1, 1))
    B2 = np.zeros((num_hidden2, 1))
    B3 = np.zeros((num_output, 1))
    dict_Param = {"W1": W1, "B1": B1, "W2": W2, "B2": B2, "W3": W3, "B3": B3}
    return dict_Param

def SaveResult(dict_param):
    np.save("Level3_w1.npy", dict_param["W1"])
    np.save("Level3_b1.npy", dict_param["B1"])
    np.save("Level3_w2.npy", dict_param["W2"])
    np.save("Level3_b2.npy", dict_param["B2"])
    np.save("Level3_w3.npy", dict_param["W3"])
    np.save("Level3_b3.npy", dict_param["B3"])

if __name__ == '__main__':

    print("Loading...")
    learning_rate = 0.2
    n_hidden1 = 64
    n_hidden2 = 32
    n_output = 10
    dataReader = LoadData(n_output)
    n_images = dataReader.num_example
    n_input = dataReader.num_feature
    m_epoch = 20
    batch_size = 100
    dict_Param = InitialParameters3(n_input, n_hidden1, n_hidden2, n_output, 2)
    dict_Param = Train(dataReader, learning_rate, m_epoch, n_images, n_input, n_output, dict_Param, forward3, backward3, update3, batch_size)
    SaveResult(dict_Param)
    Test(dataReader, n_output, dict_Param, n_input, forward3)

