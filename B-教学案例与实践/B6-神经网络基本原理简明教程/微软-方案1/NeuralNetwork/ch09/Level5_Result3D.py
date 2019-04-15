# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from LossFunction import * 
from Activators import *
from Level0_TwoLayerClassificationNet import *
from DataReader import * 
from WeightsBias import *

x_data_name = "X9_3.npy"
y_data_name = "Y9_3.npy"

def ShowAreaResult(net, wb1, wb2, title):
    count = 50
    x1 = np.linspace(0,1,count)
    x2 = np.linspace(0,1,count)
    for i in range(count):
        for j in range(count):
            x = np.array([x1[i],x2[j]]).reshape(2,1)
            dict_cache = net.ForwardCalculationBatch3(x, wb1, wb2)
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

def ShowResult3D(net, dataReader, wb1, wb2):
    X,Y,Z = PrepareData(net, dataReader, wb1, wb2)
    ShowResult(X,Y,Z[0])
    ShowResult(X,Y,Z[1])
    ShowResult(X,Y,Z[2])
    ShowResult(X,Y,Z[0]*2+Z[1])

def ShowResult(X,Y,Z):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(X,Y,Z,cmap='rainbow')
    plt.show()


def PrepareData(net, dataReader, wb1, wb2):
    count = 50
    x = np.linspace(0,1,count)
    y = np.linspace(0,1,count)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((3, len(x), len(y)))

    for i in range(count):
        for j in range(count):
            input = np.array([x[i],y[j]]).reshape(2,1)
            dict_cache = net.ForwardCalculationBatch3(input, wb1, wb2)
            Z[0,i,j] = dict_cache["Output"][0,0]
            Z[1,i,j] = dict_cache["Output"][1,0]
            Z[2,i,j] = dict_cache["Output"][2,0]
            # end if
        # end for
    # end for
    return X,Y,Z


if __name__ == '__main__':

    dataReader = DataReader(x_data_name, y_data_name)
    dataReader.ReadData()
    X = dataReader.NormalizeX()
    Y = dataReader.ToOneHot()
    
    n_input, n_output = dataReader.num_feature, dataReader.num_category
    n_hidden = 3
    eta, batch_size, max_epoch = 0.1, 10, 10000
    eps = 0.01

    params = CParameters(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, LossFunctionName.CrossEntropy3)

    loss_history = CLossHistory()
    net = TwoLayerClassificationNet()

    #ShowData(XData, YData)

    wb1, wb2 = net.train(dataReader, params, loss_history, net.ForwardCalculationBatch3)

    trace = loss_history.GetMinimalLossData()
    print(trace.toString())
    title = loss_history.ShowLossHistory(params)

    print("wait for 10 seconds...")

    ShowAreaResult(net, trace.wb1, trace.wb2, title)
    ShowData(dataReader.X, dataReader.YRawData)
    
    ShowResult3D(net, dataReader, trace.wb1, trace.wb2)