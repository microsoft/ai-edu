# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from Level0_TwoLayerClassificationNet import *
from DataReader import * 
from WeightsBias import *
from Level1_LogicXorGate import *
    
def ShowProcess2D(net, dataReader, wb1, wb2):
    dict_cache = net.ForwardCalculationBatch2(dataReader.X, wb1, wb2)
    Z1=dict_cache["Z1"]
    A1=dict_cache["A1"]
    Z2=dict_cache["Z2"]
    A2=dict_cache["A2"]

    for i in range(dataReader.num_example):
        if dataReader.Y[0,i] == 0:
            plt.plot(dataReader.X[0,i],dataReader.X[1,i],'^',c='r')
        else:
            plt.plot(dataReader.X[0,i],dataReader.X[1,i],'o',c='g')
    plt.grid()
    plt.title("X1:X2")
    plt.show()

    for i in range(dataReader.num_example):
        if dataReader.Y[0,i] == 0:
            plt.plot(Z1[0,i],Z1[1,i],'^',c='r')
        else:
            plt.plot(Z1[0,i],Z1[1,i],'o',c='g')
    plt.grid()
    plt.title("Z1")
    plt.show()

    for i in range(dataReader.num_example):
        if dataReader.Y[0,i] == 0:
            plt.plot(A1[0,i],A1[1,i],'^',c='r')
        else:
            plt.plot(A1[0,i],A1[1,i],'o',c='g')
    plt.grid()
    plt.title("A1")
    plt.show()

    x = np.linspace(-6,6)
    a = Sigmoid().forward(x)
    plt.plot(x,a)

    for i in range(dataReader.num_example):
        if dataReader.Y[0,i] == 0:
            plt.plot(Z2[0,i],A2[0,i],'^',c='r')
        else:
            plt.plot(Z2[0,i],A2[0,i],'o',c='g')
    plt.grid()
    plt.title("Z2:A2")
    plt.show()


def train():

    dataReader = XOR_DataReader()
    dataReader.ReadData()
    
    n_input, n_output = dataReader.num_feature, dataReader.num_category
    n_hidden = 2    # don't change this setting
    eta, batch_size, max_epoch = 0.1, 1, 10000
    eps = 0.005

    params = CParameters(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, LossFunctionName.CrossEntropy2)

    loss_history = CLossHistory()
    net = TwoLayerClassificationNet()

    #ShowData(XData, YData)

    wb1, wb2 = net.train(dataReader, params, loss_history, net.ForwardCalculationBatch2)

    trace = loss_history.GetMinimalLossData()
    print(trace.toString())
    title = loss_history.ShowLossHistory(params)

    print(wb1.toString())
    print(wb2.toString())

    print("wait for 10 seconds...")

    ShowResult2D(net, trace.wb1, trace.wb2, title)
    ShowData(dataReader.X, dataReader.Y)

    Test(dataReader, net, wb1, wb2)

    SaveWeights(wb1, wb2)

    return net, dataReader, wb1, wb2

def LoadWeights():
    wb1 = WeightsBias(2,2,0.1,InitialMethod.Xavier)
    wb2 = WeightsBias(1,2,0.1,InitialMethod.Xavier)
    wb1.W = np.load("xor_2d_w1.npy")
    wb1.B = np.load("xor_2d_b1.npy")
    wb2.W = np.load("xor_2d_w2.npy")
    wb2.B = np.load("xor_2d_b2.npy")
    print(wb1.toString())    
    print(wb2.toString())
    return wb1, wb2

if __name__ == '__main__':
    net,dataReader, wb1, wb2 = train()
    ShowProcess2D(net, dataReader, wb1, wb2)

