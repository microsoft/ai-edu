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
from DataOperator import * 

x_data_name = "CurveX.dat"
y_data_name = "CurveY.dat"

def ShowResult(net, X, Y, dict_weights):
    # draw train data
    plt.plot(X[0,:], Y[0,:], '.', c='b')
    # create and draw visualized validation data
    TX = np.linspace(0,1,100).reshape(1,100)
    dict_cache = net.ForwardCalculationBatch(TX, dict_weights)
    TY = dict_cache["Output"]
    plt.plot(TX, TY, 'x', c='r')
    plt.show()

if __name__ == '__main__':

    X,Y = DataOperator.ReadData(x_data_name, y_data_name)
    num_example = X.shape[1]
    n_input, n_hidden, n_output = 1, 10, 1
    eta, batch_size, max_epoch = 0.5, 5, 50000
    eps = 0.001
    init_method = InitialMethod.xavier

    params = CParameters(num_example, n_input, n_output, n_hidden, eta, max_epoch, batch_size, LossFunctionName.MSE, eps, init_method)

    # SGD, MiniBatch, FullBatch
    loss_history = CLossHistory()
    net = CTwoLayerNet()
    dict_weights = net.train(X, Y, params, loss_history)

    bookmark = loss_history.GetMinimalLossData()
    bookmark.print_info()
    loss_history.ShowLossHistory(params)

    ShowResult(net, X, Y, bookmark.weights)


