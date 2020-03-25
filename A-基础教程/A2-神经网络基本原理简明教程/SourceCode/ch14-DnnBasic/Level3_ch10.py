# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math

from MiniFramework.NeuralNet_4_0 import *
from MiniFramework.ActivationLayer import *
from MiniFramework.ClassificationLayer import *
from MiniFramework.DataReader_2_0 import *

train_data_name = "../../Data/ch10.train.npz"
test_data_name = "../../Data/ch10.test.npz"

def DrawTwoCategoryPoints(X1, X2, Y, xlabel="x1", ylabel="x2", title=None, show=False, isPredicate=False):
    colors = ['b', 'r']
    shapes = ['o', 'x']
    assert(X1.shape[0] == X2.shape[0] == Y.shape[0])
    count = X1.shape[0]
    for i in range(count):
        j = (int)(round(Y[i]))
        if j < 0:
            j = 0
        if isPredicate:
            plt.scatter(X1[i], X2[i], color=colors[j], marker='^', s=200, zorder=10)
        else:
            plt.scatter(X1[i], X2[i], color=colors[j], marker=shapes[j], zorder=10)
    # end for
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()

def ShowDataHelper(x1,x2,y,title,xlabel,ylabel,show,grid=True):
    fig = plt.figure(figsize=(6,6))
    if grid:
        plt.grid()
    DrawTwoCategoryPoints(x1,x2,y,xlabel,ylabel,title,show)

def Prepare3DData(net, count):
    x = np.linspace(0,1,count)
    y = np.linspace(0,1,count)
    X, Y = np.meshgrid(x, y)
    if net is not None:
        input = np.hstack((X.ravel().reshape(count*count,1),Y.ravel().reshape(count*count,1)))
        net.inference(input)
    return X, Y

def ShowResult2D(net, dr):
    ShowDataHelper(dr.XTrain[:,0], dr.XTrain[:,1], dr.YTrain[:,0], 
                   "Classifier Result", "x1", "x2", False, False)
    count = 50
    X,Y = Prepare3DData(net, count)
    Z = net.output.reshape(count,count)
    plt.contourf(X, Y, Z, cmap=plt.cm.Spectral, zorder=1)
    plt.show()

#end def

def load_data():
    dataReader = DataReader_2_0(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.Shuffle()
    dataReader.GenerateValidationSet()
    return dataReader

def model(dataReader):
    num_input = 2
    num_hidden = 3
    num_output = 1

    max_epoch = 1000
    batch_size = 5
    learning_rate = 0.1

    params = HyperParameters_4_0(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.BinaryClassifier,
        init_method=InitialMethod.Xavier,
        stopper=Stopper(StopCondition.StopLoss, 0.02))

    net = NeuralNet_4_0(params, "Arc")

    fc1 = FcLayer_1_0(num_input, num_hidden, params)
    net.add_layer(fc1, "fc1")
    sigmoid1 = ActivationLayer(Sigmoid())
    net.add_layer(sigmoid1, "sigmoid1")
    
    fc2 = FcLayer_1_0(num_hidden, num_output, params)
    net.add_layer(fc2, "fc2")
    logistic = ClassificationLayer(Logistic())
    net.add_layer(logistic, "logistic")

    net.train(dataReader, checkpoint=10, need_test=True)
    return net

if __name__ == '__main__':
    dr = load_data()
    net = model(dr)
    net.ShowLossHistory()
    ShowResult2D(net, dr)

