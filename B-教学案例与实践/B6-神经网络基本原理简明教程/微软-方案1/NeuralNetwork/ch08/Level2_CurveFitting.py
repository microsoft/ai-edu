# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
from LossFunction import * 
from Parameters import *
from Activators import *
from Level1_TwoLayerFitting import *
from DataReader import *

x_data_name = "CurveX.dat"
y_data_name = "CurveY.dat"

def ShowResult(net, X, Y, wb1, wb2):
    # draw train data
    plt.plot(X[0,:], Y[0,:], '.', c='b')
    # create and draw visualized validation data
    TX = np.linspace(0,1,100).reshape(1,100)
    dict_cache = net.ForwardCalculationBatch(TX, wb1, wb2)
    TY = dict_cache["Output"]
    plt.plot(TX, TY, 'x', c='r')
    plt.show()
#end def

if __name__ == '__main__':
    dataReader = DataReader(x_data_name, y_data_name)
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.NormalizeY()

    n_input, n_hidden, n_output = 1, 4, 1
    eta, batch_size, max_epoch = 0.5, 10, 50000
    eps = 0.001

    params = CParameters(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps)

    # SGD, MiniBatch, FullBatch
    loss_history = CLossHistory()
    net = TwoLayerFittingNet()
    wb1, wb2 = net.train(dataReader, params, loss_history)

    trace = loss_history.GetMinimalLossData()
    print(trace.toString())
    loss_history.ShowLossHistory(params)

    ShowResult(net, dataReader.X, dataReader.Y, trace.wb1, trace.wb2)


