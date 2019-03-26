# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
from LossFunction import * 
from Activations import *
from Level0_TwoLayerClassificationNet import *
from DataReader import * 
from GDOptimizer import *
from WeightsBias import *

x_data_name = "X9_3.npy"
y_data_name = "Y9_3.npy"

def ShowData(X, Y):
    for i in range(X.shape[1]):
        if Y[0,i] == 1:
            plt.plot(X[0,i], X[1,i], '^', c='g')
        elif Y[0,i] == 2:
            plt.plot(X[0,i], X[1,i], 'x', c='r')
        elif Y[0,i] == 3:
            plt.plot(X[0,i], X[1,i], '.', c='b')
        # end if
    # end for
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


if __name__ == '__main__':

    dataReader = DataReader(x_data_name, y_data_name)
    XData,YData = dataReader.ReadData()
    X = dataReader.NormalizeX()
    Y = dataReader.ToOneHot()
    
    n_input, n_output = dataReader.num_feature, dataReader.num_category
    n_hidden = 8
    eta, batch_size, max_epoch = 0.1, 10, 10000
    eps = 0.05

    params = CParameters(n_input, n_output, n_hidden,
                         eta, max_epoch, batch_size, eps, 
                         LossFunctionName.CrossEntropy3, 
                         InitialMethod.Xavier,
                         OptimizerName.SGD)

    loss_history = CLossHistory()
    net = TwoLayerClassificationNet()

    ShowData(XData, YData)

    net.train(dataReader, params, loss_history)

    trace = loss_history.GetMinimalLossData()
    trace.toString()
    loss_history.ShowLossHistory(params)
    