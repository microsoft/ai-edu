# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from Level3_CurveFitting import *

def ReadData():
    x1 = np.random.random((1,100)) * 2 * 3.14
    x2 = np.sin(x1)
    X = np.zeros((2,100))
    X[0] = x1
    X[1] = x2
    Y = 0.2 * x1 + 1
    return X,Y

def ShowResult(X,dict):
    plt.plot(X[0,:], X[1,:], '.')
    cache = ForwardCalculationBatch(X, dict)
    plt.plot(X[0,:].reshape(1,-1), cache["A2"], 'x', c='r')
    plt.show()

if __name__ == '__main__':
    
    X,Y = ReadData()
    num_example = X.shape[1]
    n_input, n_hidden, n_output = 2, 4, 1
    eta, batch_size, max_epoch = 0.2, 10, 1000
    eps = 0.001
    params = CParameters(num_example, n_input, n_output, n_hidden, eta, max_epoch, batch_size, "MSE", eps)
    loss_history = CLossHistory()
    train(X, Y, params, loss_history)

    bookmark = loss_history.GetMinimalLossData()
    bookmark.print_info()
    loss_history.ShowLossHistory(params)

    ShowResult(X, bookmark.weights)
    print(bookmark.weights["W1"])
    print(bookmark.weights["B1"])
    print(bookmark.weights["W2"])
    print(bookmark.weights["B2"])

