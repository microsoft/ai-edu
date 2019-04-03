# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from LossFunction import * 
from Parameters import *
from Activators import *
from Level1_TwoLayerFittingNet import *
from DataReader import *

x_data_name = "X8.dat"
y_data_name = "Y8.dat"

def ShowResult(net, X, Y, title, wb1, wb2):
    # draw train data
    plt.plot(X[0,:], Y[0,:], '.', c='b')
    # create and draw visualized validation data
    TX = np.linspace(0,1,100).reshape(1,100)
    dict_cache = net.ForwardCalculationBatch(TX, wb1, wb2)
    TY = dict_cache["Output"]
    plt.plot(TX, TY, 'x', c='r')
    plt.title(title)
    plt.show()
#end def

def ShowResult3D(net, X, Y, title, wb1, wb2):
    TX = np.linspace(0,1,10).reshape(1,10)
    dict_cache = net.ForwardCalculationBatch(TX, wb1, wb2)
    Z1 = dict_cache["Z1"]
    A1 = dict_cache["A1"]
    Z2 = dict_cache["Z2"]
    '''
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(Z1[0],Z1[1],Z1[2],c='g')
    ax.scatter(A1[0],A1[1],A1[2],c='r')
    plt.show()
    '''
    plt.plot(TX[0],Z2[0])
    plt.show()

def Train():
    dataReader = DataReader(x_data_name, y_data_name)
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.NormalizeY()

    n_input, n_hidden, n_output = 1, 3, 1
    eta, batch_size, max_epoch = 0.5, 10, 50000
    eps = 0.001

    params = CParameters(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps)

    # SGD, MiniBatch, FullBatch
    loss_history = CLossHistory()
    net = TwoLayerFittingNet()
    wb1, wb2 = net.train(dataReader, params, loss_history)

    trace = loss_history.GetMinimalLossData()
    print(trace.toString())
    title = loss_history.ShowLossHistory(params)

    ShowResult(net, dataReader.X, dataReader.Y, title, trace.wb1, trace.wb2)
    trace.wb1.Save("wb1")
    trace.wb2.Save("wb2")


if __name__ == '__main__':
    # Train()
    dataReader = DataReader(x_data_name, y_data_name)
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.NormalizeY()

    n_input, n_hidden, n_output = 1, 3, 1
    eta, batch_size, max_epoch = 0.5, 10, 50000
    eps = 0.001

    params = CParameters(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps)

    wb1 = WeightsBias(n_input, n_hidden, eta)
    wb2 = WeightsBias(n_hidden, n_output, eta)
    wb1.Load("wb1")
    wb2.Load("wb2")

    # SGD, MiniBatch, FullBatch
    loss_history = CLossHistory()
    net = TwoLayerFittingNet()
    #ShowResult(net, dataReader.X, dataReader.Y, "title", wb1, wb2)
    ShowResult3D(net, dataReader.X, dataReader.Y, "title", wb1, wb2)
