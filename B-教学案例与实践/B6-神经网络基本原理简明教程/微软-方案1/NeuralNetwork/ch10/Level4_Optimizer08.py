# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math

from LossFunction import * 
from Level0_TwoLayerNet import *
from DataReader import * 
from GDOptimizer import *
from WeightsBias import *

x_data_name = "X8.dat"
y_data_name = "Y8.dat"

def ShowResult(net, X, Y, title, wb1, wb2):
    # draw train data
    plt.plot(X[0,:], Y[0,:], '.', c='b')
    # create and draw visualized validation data
    TX = np.linspace(0,1,100).reshape(1,100)
    dict_cache = net.forward(TX, wb1, wb2)
    TY = dict_cache["Output"]
    plt.plot(TX, TY, 'x', c='r')
    plt.title(title)
    plt.show()
#end def


def WalkThroughAllOptimizers(option):

    dataReader = DataReader(x_data_name, y_data_name)
    XData,YData = dataReader.ReadData()
    X = dataReader.NormalizeX()
    Y = dataReader.NormalizeY()
    
    n_input, n_output = dataReader.num_feature, 1
    n_hidden = option[2]
    eta, batch_size, max_epoch = option[1], 10, 10000
    eps = 0.001

    params = CParameters(n_input, n_hidden, n_output,
                         eta, max_epoch, batch_size, eps, 
                         InitialMethod.Xavier,
                         option[0])

    loss_history = CLossHistory()
    net = TwoLayerNet(NetType.Fitting)

    wbs = net.train(dataReader, params, loss_history)

    trace = loss_history.GetMinimalLossData()
    print(trace.toString())
    title = loss_history.ShowLossHistory(params)

    print("wait for 10 seconds...")

    ShowResult(net, X, Y, title, trace.wb1, trace.wb2)
   
if __name__ == '__main__':

    WalkThroughAllOptimizers((OptimizerName.AdaDelta,0.1,2))
    WalkThroughAllOptimizers((OptimizerName.Adam,0.001,2))

    '''
    list_name = [(OptimizerName.SGD,0.1,4),
                 (OptimizerName.Momentum,0.1,3),
                 (OptimizerName.Momentum,0.1,4),
                 (OptimizerName.Nag,0.1,3),
                 (OptimizerName.Nag,0.1,4),
                 (OptimizerName.AdaGrad,0.3,4),
                 (OptimizerName.AdaGrad,0.5,4),
                 (OptimizerName.AdaGrad,0.7,4),
                 (OptimizerName.AdaDelta,0.1,4),
                 (OptimizerName.AdaDelta,0.01,4),
                 (OptimizerName.RMSProp,0.1,4),
                 (OptimizerName.RMSProp,0.01,4),
                 (OptimizerName.RMSProp,0.005,4),
                 (OptimizerName.Adam,0.1,4),
                 (OptimizerName.Adam,0.01,4),
                 (OptimizerName.Adam,0.005,4),
                 (OptimizerName.Adam,0.001,4)
                 ]

    for name in list_name:
        WalkThroughAllOptimizers(name)
    '''