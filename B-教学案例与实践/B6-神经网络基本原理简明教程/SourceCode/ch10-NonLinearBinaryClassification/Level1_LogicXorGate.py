# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math

from LossFunction import * 
from Activators import *
from Level0_TwoLayerClassificationNet import *
from DataReader import * 
from WeightsBias import *


# x1=0,x2=0,y=0
# x1=0,x2=1,y=1
# x1=1,x2=0,y=1
# x1=1,x2=1,y=0
class XOR_DataReader():
    def __init__(self):
        pass

    def ReadData(self):
        self.X = np.array([0,0,1,1,0,1,0,1]).reshape(2,4)
        self.Y = np.array([0,1,1,0]).reshape(1,4)
        self.num_example = self.X.shape[1]
        self.num_feature = self.X.shape[0]
        self.num_category = 1

    def GetBatchSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_X = self.X[0:self.num_feature, start:end].reshape(self.num_feature, batch_size)
        batch_Y = self.Y[:, start:end].reshape(-1, batch_size)
        return batch_X, batch_Y


def ShowResult2D(net, wb1, wb2, title):
    count = 50
    x1 = np.linspace(0,1,count)
    x2 = np.linspace(0,1,count)
    for i in range(count):
        for j in range(count):
            x = np.array([x1[i],x2[j]]).reshape(2,1)
            dict_cache = net.forward(x, wb1, wb2)
            output = dict_cache["Output"]
            if output[0,0] >= 0.5:
                plt.plot(x[0,0], x[1,0], 's', c='m')
            else:
                plt.plot(x[0,0], x[1,0], 's', c='y')
            # end if
        # end for
    # end for
    plt.title(title)
#end def

def ShowData(X, Y):
    for i in range(X.shape[1]):
        if Y[0,i] == 0:
            plt.plot(X[0,i], X[1,i], '^', c='r')
        elif Y[0,i] == 1:
            plt.plot(X[0,i], X[1,i], '.', c='g')
        # end if
    # end for
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

def Test(dataReader, net, wb1, wb2):
    print("testing...")
    for i in range(dataReader.num_example):
        x,y = dataReader.GetBatchSamples(1, i)
        dict_output = net.forward(x, wb1, wb2)
        output = dict_output["Output"]
        print(str.format("x={0} y={1} output={2}", x, y, output))
        if np.abs(output - y) < 1e-2:
            print("True")
        else:
            print("False")
        #end if
    #end for

def SaveWeights(wb1, wb2):
    np.save("xor_2d_w1.npy", wb1.W)
    np.save("xor_2d_b1.npy", wb1.B)
    np.save("xor_2d_w2.npy", wb2.W)
    np.save("xor_2d_b2.npy", wb2.B)


def train():

    dataReader = XOR_DataReader()
    dataReader.ReadData()
    
    n_input, n_output = dataReader.num_feature, 1
    n_hidden = 2
    eta, batch_size, max_epoch = 0.1, 1, 10000
    eps = 0.01

    params = CParameters(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, LossFunctionName.CrossEntropy2)

    loss_history = CLossHistory()
    net = TwoLayerClassificationNet()

    #ShowData(XData, YData)

    wb1, wb2 = net.train(dataReader, params, loss_history)

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

if __name__ == '__main__':
    train()