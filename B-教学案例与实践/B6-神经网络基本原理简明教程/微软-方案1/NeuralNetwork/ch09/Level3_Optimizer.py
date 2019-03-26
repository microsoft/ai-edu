# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
from LossFunction import * 
from Activations import *
from Level0_TwoLayerFittingNet import *
from DataReader import * 
from GDOptimizer import *
from WeightsBias import *

x_data_name = "X8.npy"
y_data_name = "Y8.npy"

def ShowResult(X, Y, net, wbs, title):
    # draw train data
    plt.plot(X[0,:], Y[0,:], '.', c='b')
    # create and draw visualized validation data
    TX = np.linspace(0,1,100).reshape(1,100)
    dict_cache = net.ForwardCalculationBatch(TX, wbs)
    TY = dict_cache["Output"]
    plt.plot(TX, TY, 'x', c='r')
    plt.title(title)
    plt.show()
#end def


def WalkThroughAllOptimizers(optname):

    dataReader = DataReader(x_data_name, y_data_name)
    XData,YData = dataReader.ReadData()
    X = dataReader.NormalizeX()
    Y = dataReader.NormalizeY()
    
    n_input, n_output = dataReader.num_feature, 1
    n_hidden = 4
    eta, batch_size, max_epoch = 0.1, 10, 20000
    eps = 0.001

    params = CParameters(n_input, n_output, n_hidden,
                         eta, max_epoch, batch_size, eps, 
                         LossFunctionName.MSE, 
                         InitialMethod.Xavier,
                         optname)

    loss_history = CLossHistory()
    net = TwoLayerFittingNet()

    #ShowData(XData, YData)

    wbs = net.train(dataReader, params, loss_history)

    trace = loss_history.GetMinimalLossData()
    print(trace.toString())
    title = loss_history.ShowLossHistory(params)

    print("wait for 10 seconds...")

    wbs_min = WeightsBias(params)
    wbs_min.W1 = trace.dict_weights["W1"]
    wbs_min.W2 = trace.dict_weights["W2"]
    wbs_min.B1 = trace.dict_weights["B1"]
    wbs_min.B2 = trace.dict_weights["B2"]
    ShowResult(X, Y, net, wbs_min, title)
   
if __name__ == '__main__':

    list_name = [OptimizerName.SGD, 
                 OptimizerName.Momentum,
                 OptimizerName.Nag,
                 OptimizerName.AdaGrad,
                 OptimizerName.RMSProp,
                 OptimizerName.Adam]

    for name in list_name:
        WalkThroughAllOptimizers(name)