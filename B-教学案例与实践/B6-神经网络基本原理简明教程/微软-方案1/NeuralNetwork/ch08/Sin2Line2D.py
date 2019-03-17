# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Level3_CurveFitting import *

def ReadData():
    x1 = np.random.random((1,100)) * 2 * 3.14
    x2 = np.sin(x1)
    X = np.zeros((2,100))
    X[0] = x1
    X[1] = x2
    Y = 1 * x1 + 1
    return X,Y

def ShowResult(X,dict):
    plt.plot(X[0,:], X[1,:], '.')
    cache = ForwardCalculationBatch(X, dict)
    plt.plot(X[0,:].reshape(1,-1), cache["A2"], 'x', c='r')
    plt.show()

if __name__ == '__main__':
    
    X,Y = ReadData()
    num_example = X.shape[1]
    n_input, n_hidden, n_output = 2, 2, 1
    # 0.1, 10, 50000
    # 0.01, 20, 50000
    eta, batch_size, max_epoch = 0.05, 10, 50000
    eps = 0.0001
    init_method = 2
    params = CParameters(num_example, n_input, n_output, n_hidden, eta, max_epoch, batch_size, "MSE", eps, init_method)
    loss_history = CLossHistory()
    dict_weights = train(X, Y, params, loss_history)

    bookmark = loss_history.GetMinimalLossData()
    bookmark.print_info()
    loss_history.ShowLossHistory(params)

 #   ShowResult(X, bookmark.weights)
    print(bookmark.weights["W1"])
    print(bookmark.weights["B1"])
    print(bookmark.weights["W2"])
    print(bookmark.weights["B2"])

    W1 = dict_weights["W1"]
    B1 = dict_weights["B1"]
    W2 = dict_weights["W2"]
    B2 = dict_weights["B2"]
    Z = np.dot(W1, X) + B1
    A = CSigmoid().forward(Z)
    # for n_hidden = 2
    Z2 = np.dot(W2, A) + B2
    Z3 = np.dot(W2, Z) + B2

    # source
    plt.plot(X[0,:], X[1,:], '.', c='r')
    plt.plot(Z[0,:], Z[1,:], '.', c='g')
    plt.plot(X[0,:], Z2[0,:], '.', c='b')
    plt.plot(X[0,:], Z3[0,:], '.', c='y')
     
    plt.show()
