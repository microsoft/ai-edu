# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import struct
import matplotlib.pyplot as plt

from Level1_Base import *

train_image_file = 'train-images-10'
train_label_file = 'train-labels-10'
test_image_file = 'test-images-10'
test_label_file = 'test-labels-10'

def Forward(X, dict_Param):
    W1 = dict_Param["W1"]
    B1 = dict_Param["B1"]
    W2 = dict_Param["W2"]
    B2 = dict_Param["B2"]
    
    Z1 = np.dot(W1,X)+B1
    A1 = Sigmoid(Z1)

    Z2=np.dot(W2,A1)+B2
    A2=Softmax(Z2)
    
    dict_Cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Output": A2}
    return dict_Cache

def Backward(dict_Param,cache,X,Y):
    W1=dict_Param["W1"]
    W2=dict_Param["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    m = X.shape[1]

    dZ2= A2 - Y
    dW2 = np.dot(dZ2, A1.T)/m
    dB2 = np.sum(dZ2, axis=1, keepdims=True)/m

    dLoss_A1 = np.dot(W2.T, dZ2)
    dA1_Z1 = A1 * (1 - A1)     # sigmoid
    dZ1 = dLoss_A1 * dA1_Z1
    
    dW1 = np.dot(dZ1, X.T)/m
    dB1 = np.sum(dZ1, axis=1, keepdims=True)/m

    dict_Grads = {"dW1": dW1, "dB1": dB1, "dW2": dW2, "dB2": dB2}
    return dict_Grads

def Update(dict_Param, dict_Grads, learning_rate):
    W1 = dict_Param["W1"]
    B1 = dict_Param["B1"]
    W2 = dict_Param["W2"]
    B2 = dict_Param["B2"]

    dW1 = dict_Grads["dW1"]
    dB1 = dict_Grads["dB1"]
    dW2 = dict_Grads["dW2"]
    dB2 = dict_Grads["dB2"]

    W1 = W1 - learning_rate * dW1
    B1 = B1 - learning_rate * dB1
    W2 = W2 - learning_rate * dW2
    B2 = B2 - learning_rate * dB2

    dict_Param = {"W1": W1, "B1": B1, "W2": W2, "B2": B2}
    return dict_Param


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


if __name__ == '__main__':
    print("Loading...")
    learning_rate = 0.2
    n_hidden = 32
    n_output = 10
    dataReader = LoadData(n_output)
    n_images = dataReader.num_example
    n_input = dataReader.num_feature
    m_epoch = 20
    batch_size = 100
    dict_Param = InitialParameters(n_input, n_hidden, n_output, 2)
    dict_Param = Train(dataReader, learning_rate, m_epoch, n_images, n_input, n_output, dict_Param, Forward, Backward, Update, batch_size)
    Test(dataReader, n_output, dict_Param, n_input, Forward)
    