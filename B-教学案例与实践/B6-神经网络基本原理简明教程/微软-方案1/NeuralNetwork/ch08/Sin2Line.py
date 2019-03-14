# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from MiniBatch import *

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
    loss_history = CLossHistory()
    # SGD, MiniBatch, FullBatch
    method = "SGD"

    n_input, n_hidden, n_output = 2, 4, 1

    X,Y = ReadData()
    num_samples = X.shape[1]
    train(method, X, Y, n_input, n_hidden, n_output, loss_history)

    bookmark = loss_history.GetMinimalLossData()
    print("epoch=%d, iteration=%d, loss=%f" %(bookmark.epoch, bookmark.iteration, bookmark.loss))
    loss_history.ShowLossHistory(method)

    ShowResult(X, bookmark.weights)
    print(bookmark.weights["W1"])
    print(bookmark.weights["B1"])
    print(bookmark.weights["W2"])
    print(bookmark.weights["B2"])


