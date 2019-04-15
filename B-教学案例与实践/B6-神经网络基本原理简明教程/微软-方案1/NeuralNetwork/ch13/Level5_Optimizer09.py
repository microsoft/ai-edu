# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
from LossFunction import * 
from Activations import *
from Level1_TwoLayerClassificationNet import *
from DataReader import * 
from GDOptimizer import *
from WeightsBias import *

x_data_name = "X9_3.npy"
y_data_name = "Y9_3.npy"

def ShowAreaResult(X, wbs, net, title):
    count = 50
    x1 = np.linspace(0,1,count)
    x2 = np.linspace(0,1,count)
    for i in range(count):
        for j in range(count):
            x = np.array([x1[i],x2[j]]).reshape(2,1)
            dict_cache = net.ForwardCalculationBatch(x, wbs)
            output = dict_cache["Output"]
            r = np.argmax(output, axis=0)
            if r == 0:
                plt.plot(x[0,0], x[1,0], 's', c='m')
            elif r == 1:
                plt.plot(x[0,0], x[1,0], 's', c='y')
            # end if
        # end for
    # end for
    plt.title(title)
#end def

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


def WalkThroughAllOptimizers(option):

    dataReader = DataReader(x_data_name, y_data_name)
    XData,YData = dataReader.ReadData()
    X = dataReader.NormalizeX()
    Y = dataReader.ToOneHot()
    
    n_input, n_output = dataReader.num_feature,  dataReader.num_category
    n_hidden = 8
    eta, batch_size, max_epoch = option[1], 10, 10000
    eps = 0.06

    params = CParameters(n_input, n_output, n_hidden,
                         eta, max_epoch, batch_size, eps, 
                         LossFunctionName.CrossEntropy3, 
                         InitialMethod.Xavier,
                         option[0])

    loss_history = CLossHistory()
    net = TwoLayerClassificationNet()

    #ShowData(XData, YData)

    net.train(dataReader, params, loss_history)

    trace = loss_history.GetMinimalLossData()
    print(trace.toString())
    title = loss_history.ShowLossHistory(params)

    print("wait for 10 seconds...")

    wbs_min = WeightsBias(params)
    wbs_min.W1 = trace.dict_weights["W1"]
    wbs_min.W2 = trace.dict_weights["W2"]
    wbs_min.B1 = trace.dict_weights["B1"]
    wbs_min.B2 = trace.dict_weights["B2"]
    ShowAreaResult(X, wbs_min, net, title)
    ShowData(X, YData)

   
if __name__ == '__main__':

    list_name = [(OptimizerName.SGD,0.1),
                 (OptimizerName.Momentum,0.1),
                 (OptimizerName.Nag,0.1),
                 (OptimizerName.AdaGrad,0.3),
                 (OptimizerName.AdaDelta,0.1),
                 (OptimizerName.RMSProp,0.01),
                 (OptimizerName.Adam,0.005)]

    for name in list_name:
        WalkThroughAllOptimizers(name)