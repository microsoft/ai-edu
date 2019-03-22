# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
from LossFunction import * 
from Parameters import *
from Activations import *
from Level1_TwoLayer import *

x_data_name = "X9_3.npy"
y_data_name = "Y9_3.npy"

class CAdaGrad(object):
    def __init__(self, eta):
        self.delta = 1e-7
        self.eta = eta
        self.r = 0

    def step(self, theta, grad):
        self.r = self.r + np.multiply(grad, grad)
        alpha = self.eta / (np.sqrt(self.r) + self.delta)
        #print("alpha=",alpha)
        d_theta = -np.multiply(alpha, grad)
        theta = theta + d_theta
        return theta

class CRMSProp(object):
    def __init__(self, eta):
        self.eta = eta
        self.p = 0.9
        self.delta = 1e-6
        self.r = 0

    def step(self, theta, grad):
        grad2 = np.multiply(grad, grad)
        self.r = self.p * self.r + (1-self.p) * grad2
        alpha = self.eta / np.sqrt(self.delta + self.r)
        theta = theta - np.multiply(alpha, grad)
        return theta

class CAdam(object):
    def __init__(self, shape):
        self.eta = 0.1
        self.p1 = 0.9
        self.p2 = 0.999
        self.delta = 1e-8
        self.s = np.zeros(shape)
        self.r = np.zeros(shape)
        self.t = 0

    def step(self, theta, grad):
        self.t = self.t + 1
        self.s = self.p1 * self.s + (1-self.p1) * grad
        self.r = self.p2 * self.r + (1-self.p2) * np.multiply(grad, grad)
        s1 = self.s / (1 - self.p1 ** self.t)
        r1 = self.r / (1 - self.p2 ** self.t)
        alpha = self.eta * s1 / (self.delta + np.sqrt(r1))
        theta = theta - alpha
        return theta

class COptimizer(CTwoLayerNet):
    def UpdateWeights(self, dict_weights, dict_grads, dict_optimizer):
        W1 = dict_weights["W1"]
        B1 = dict_weights["B1"]
        W2 = dict_weights["W2"]
        B2 = dict_weights["B2"]

        dW1 = dict_grads["dW1"]
        dB1 = dict_grads["dB1"]
        dW2 = dict_grads["dW2"]
        dB2 = dict_grads["dB2"]

        W1 = dict_optimizer["W1"].step(W1, dW1)
        B1 = dict_optimizer["B1"].step(B1, dB1)
        W2 = dict_optimizer["W2"].step(W2, dW2)
        B2 = dict_optimizer["B2"].step(B2, dB2)

        dict_weights = {"W1": W1,"B1": B1,"W2": W2,"B2": B2}

        return dict_weights

    def train(self, X, Y, params, loss_history):
        num_example = X.shape[1]
        num_feature = X.shape[0]
        num_category = Y.shape[0]

        # W(num_category, num_feature), B(num_category, 1)
        W1, B1, W2, B2 = params.LoadSameInitialParameters()
        dict_weights = {"W1":W1, "B1":B1, "W2":W2, "B2":B2}
        #dict_optimizer = {"W1":CAdaGrad(params.eta), "B1":CAdaGrad(params.eta), "W2":CAdaGrad(params.eta), "B2":CAdaGrad(params.eta)}
        #dict_optimizer = {"W1":CRMSProp(params.eta), "B1":CRMSProp(params.eta), "W2":CRMSProp(params.eta), "B2":CRMSProp(params.eta)}
        dict_optimizer = {"W1":CAdam(W1.shape), "B1":CAdam(B1.shape), "W2":CAdam(W2.shape), "B2":CAdam(B2.shape)}

        # calculate loss to decide the stop condition
        loss = 0 
        lossFunc = CLossFunction(params.loss_func_name)

        # if num_example=200, batch_size=10, then iteration=200/10=20
        max_iteration = (int)(params.num_example / params.batch_size)
        for epoch in range(params.max_epoch):
            for iteration in range(max_iteration):
                # get x and y value for one sample
                batch_x, batch_y = DataOperator.GetBatchSamples(X,Y,params.batch_size,iteration)
                # get z from x,y
                dict_cache = self.ForwardCalculationBatch(batch_x, dict_weights)
                # calculate gradient of w and b
                dict_grads = self.BackPropagationBatch(batch_x, batch_y, dict_cache, dict_weights)
                # update w,b
                dict_weights = self.UpdateWeights(dict_weights, dict_grads, dict_optimizer)
            # end for            
            # calculate loss for this batch
            loss = lossFunc.CheckLoss(X, Y, dict_weights, self.ForwardCalculationBatch)
            print("epoch=%d, loss=%f" %(epoch,loss))
            loss_history.AddLossHistory(loss, dict_weights, epoch, iteration)            
            if math.isnan(loss):
                break
            # end if
            if loss < params.eps:
                break
            # end if
        # end for
        return dict_weights

if __name__ == '__main__':

    XData,YData = DataOperator.ReadData(x_data_name, y_data_name)
    norm = DataOperator("min_max")
    X = norm.NormalizeData(XData)
    num_category = 3
    Y = DataOperator.ToOneHot(YData, num_category)

    num_example = X.shape[1]
    num_feature = X.shape[0]
    
    n_input, n_hidden, n_output = num_feature, 8, num_category
    eta, batch_size, max_epoch = 0.001, 10, 10000
    eps = 0.05
    init_method = InitialMethod.xavier

    params = CParameters(num_example, n_input, n_output, n_hidden, eta, max_epoch, batch_size, LossFunctionName.CrossEntropy3, eps, init_method)

    loss_history = CLossHistory()
    optimizer = COptimizer()
    dict_weights = optimizer.train(X, Y, params, loss_history)

    bookmark = loss_history.GetMinimalLossData()
    bookmark.print_info()
    loss_history.ShowLossHistory(params)

    optimizer.ShowAreaResult(X, bookmark.weights)
    optimizer.ShowData(X, YData)
