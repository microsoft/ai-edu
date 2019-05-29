# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math

from MiniFramework.NeuralNet import *
from MiniFramework.Optimizer import *
from MiniFramework.LossFunction import *
from MiniFramework.Parameters import *
from MiniFramework.WeightsBias import *
from MiniFramework.ActivatorLayer import *
from MiniFramework.DataReader import *

x_data_name = "X11.dat"
y_data_name = "Y11.dat"

def LoadData():
    dataReader = DataReader(x_data_name, y_data_name)
    dataReader.ReadData()
    dataReader.Normalize(normalize_x = True, normalize_y = True, to_one_hot = True)
    return dataReader

def ShowData(dataReader):
    for i in range(dataReader.X.shape[1]):
        if dataReader.Y[0,i] == 1:
            plt.plot(dataReader.X[0,i], dataReader.X[1,i], '^', c='g')
        elif dataReader.Y[1,i] == 1:
            plt.plot(dataReader.X[0,i], dataReader.X[1,i], 'x', c='r')
        elif dataReader.Y[2,i] == 1:
            plt.plot(dataReader.X[0,i], dataReader.X[1,i], '.', c='b')
        # end if
    # end for
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

def ShowResult(net, title):
    print("waiting for 10 seconds...")
    count = 50
    x1 = np.linspace(0,1,count)
    x2 = np.linspace(0,1,count)
    for i in range(count):
        for j in range(count):
            x = np.array([x1[i],x2[j]]).reshape(2,1)
            output = net.inference(x)
            r = np.argmax(output, axis=0)
            if r == 0:
                plt.plot(x[0,0], x[1,0], 's', c='m')
            elif r == 1:
                plt.plot(x[0,0], x[1,0], 's', c='y')
            # end if
        # end for
    # end for
    plt.title(title)
    #plt.show()

if __name__ == '__main__':
    dataReader = LoadData()
    num_input = dataReader.num_feature
    num_hidden1 = 8
    num_output = 3

    max_epoch = 1000
    batch_size = 10
    learning_rate = 0.1
    eps = 0.06

    params = CParameters(learning_rate, max_epoch, batch_size, eps,
                        LossFunctionName.CrossEntropy3, 
                        InitialMethod.Xavier, 
                        OptimizerName.SGD)

    net = NeuralNet(params)
    fc1 = FcLayer(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    relu1 = ActivatorLayer(Relu())
    net.add_layer(relu1, "relu1")
    fc2 = FcLayer(num_hidden1, num_output, params)
    net.add_layer(fc2, "fc2")
    softmax1 = ClassificationLayer(Softmax())
    net.add_layer(softmax1, "softmax1")

    net.train(dataReader, checkpoint=10, need_test=False)
    net.ShowLossHistory()
    
    ShowResult(net, params.toString())
    ShowData(dataReader)
