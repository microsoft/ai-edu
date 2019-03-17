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

def show_3d_surface(x, y, z):
    fig = plt.figure()
    ax = Axes3D(fig)

    plt.plot(x,y,z,'.',c='black')
    plt.show()

if __name__ == '__main__':
    
    X,Y = ReadData()
    num_example = X.shape[1]
    n_input, n_hidden, n_output = 2, 3, 1
    eta, batch_size, max_epoch = 0.1, 1, 20000
    eps = 0.00001
    init_method = 1
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
    # for n_hidden = 3
    show_3d_surface(Z[0,:],Z[1,:],Z[2,:])
    show_3d_surface(A[0,:],A[1,:],A[2,:])
    Z2 = np.dot(W2, A) + B2
    plt.plot(X[0,:], Z2[0,:], 'x', c='y')


     
    plt.show()
