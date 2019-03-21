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

x_data_name = "CurveX.dat"
y_data_name = "CurveY.dat"

class CMomentum(object):
    def __init__(self, eta):
        self.vt_1 = 0
        self.eta = eta
        self.gamma = 0.9

    def step(self, theta, grad):
        vt = self.gamma * self.vt_1 - self.eta * grad
        theta = theta + vt
        self.vt_1 = vt
        return theta


class CMomentumOptimizer(CTwoLayerNet):
    def UpdateWeights(self, dict_weights, dict_grads, dict_momentum):
        W1 = dict_weights["W1"]
        B1 = dict_weights["B1"]
        W2 = dict_weights["W2"]
        B2 = dict_weights["B2"]

        dW1 = dict_grads["dW1"]
        dB1 = dict_grads["dB1"]
        dW2 = dict_grads["dW2"]
        dB2 = dict_grads["dB2"]

        W1 = dict_momentum["W1"].step(W1, dW1)
        B1 = dict_momentum["B1"].step(B1, dB1)
        W2 = dict_momentum["W2"].step(W2, dW2)
        B2 = dict_momentum["B2"].step(B2, dB2)

        dict_weights = {"W1": W1,"B1": B1,"W2": W2,"B2": B2}

        return dict_weights

    def train(self, X, Y, params, loss_history):
        num_example = X.shape[1]
        num_feature = X.shape[0]
        num_category = Y.shape[0]

        # W(num_category, num_feature), B(num_category, 1)
        W1, B1, W2, B2 = params.LoadSameInitialParameters()
        dict_weights = {"W1":W1, "B1":B1, "W2":W2, "B2":B2}
        dict_momentum = {"W1":CMomentum(params.eta), "B1":CMomentum(params.eta), "W2":CMomentum(params.eta), "B2":CMomentum(params.eta)}

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
                dict_weights = self.UpdateWeights(dict_weights, dict_grads, dict_momentum)
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

    X,Y = DataOperator.ReadData(x_data_name, y_data_name)
    num_example = X.shape[1]
    n_input, n_hidden, n_output = 1, 4, 1
    eta, batch_size, max_epoch = 0.1, 10, 10000
    eps = 0.001
    init_method = InitialMethod.xavier

    params = CParameters(num_example, n_input, n_output, n_hidden, eta, max_epoch, batch_size, LossFunctionName.MSE, eps, init_method)

    # SGD, MiniBatch, FullBatch
    loss_history = CLossHistory()
    optimizer = CMomentumOptimizer()
    dict_weights = optimizer.train(X, Y, params, loss_history)

    bookmark = loss_history.GetMinimalLossData()
    bookmark.print_info()
    loss_history.ShowLossHistory(params)

    optimizer.ShowResult(X, Y, bookmark.weights)
    print(bookmark.weights["W1"])
    print(bookmark.weights["B1"])
    print(bookmark.weights["W2"])
    print(bookmark.weights["B2"])

