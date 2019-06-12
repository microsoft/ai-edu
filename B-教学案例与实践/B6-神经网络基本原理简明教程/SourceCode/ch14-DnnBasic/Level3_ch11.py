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

train_file = "../../Data/11_Train.npz"
test_file = "../../Data/11_Test.npz"

def LoadData():
    dr = DataReader(train_file, test_file)
    dr.ReadData()
    dr.NormalizeX()
    dr.NormalizeY(YNormalizationMethod.MultipleClassifier, base=1)
    dr.Shuffle()
    dr.GenerateValidationSet()
    return dr

def ShowData(dataReader):
    for i in range(dataReader.XTrain.shape[0]):
        if dataReader.YTrain[i,0] == 1:
            plt.plot(dataReader.XTrain[i,0], dataReader.XTrain[i,1], '^', c='g')
        elif dataReader.YTrain[i,1] == 1:
            plt.plot(dataReader.XTrain[i,0], dataReader.XTrain[i,1], 'x', c='r')
        elif dataReader.YTrain[i,2] == 1:
            plt.plot(dataReader.XTrain[i,0], dataReader.XTrain[i,1], '.', c='b')
        # end if
    # end for
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

def ShowResult(net, title):
    fig = plt.figure(figsize=(5,5))
    count = 50
    x = np.linspace(0,1,count)
    y = np.linspace(0,1,count)
    X,Y = np.meshgrid(x,y)
    z = net.inference(np.c_[X.ravel(),Y.ravel()])
    Z = np.max(z,axis=1).reshape(X.shape)
    plt.contourf(X,Y,Z)

def model():
    dataReader = LoadData()
    num_input = dataReader.num_feature
    num_hidden1 = 8
    num_output = 3

    max_epoch = 5000
    batch_size = 10
    learning_rate = 0.1
    eps = 0.06

    params = CParameters(
        learning_rate, max_epoch, batch_size, eps,
        LossFunctionName.CrossEntropy3, 
        InitialMethod.Xavier, 
        OptimizerName.SGD)

    net = NeuralNet(params, "chinabank")

    fc1 = FcLayer(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    r1 = ActivatorLayer(Relu())
    net.add_layer(r1, "Relu1")

    fc2 = FcLayer(num_hidden1, num_output, params)
    net.add_layer(fc2, "fc2")
    softmax1 = ClassificationLayer(Softmax())
    net.add_layer(softmax1, "softmax1")

    net.train(dataReader, checkpoint=10, need_test=True)
    net.ShowLossHistory()
    
    ShowResult(net, params.toString())
    ShowData(dataReader)

if __name__ == '__main__':
    model()