# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

from HelperClass2.NeuralNet_2_1 import *
from Level2_XorGateHow import *

train_data_name = "../../Data/ch10.train.npz"
test_data_name = "../../Data/ch10.test.npz"

def DrawSamplePoints(x1, x2, y, title, xlabel, ylabel, show=True):
    assert(x1.shape[0] == x2.shape[0])
    fig = plt.figure(figsize=(6,6))
    count = x1.shape[0]
    for i in range(count):
        if y[i,0] == 0:
            plt.scatter(x1[i], x2[i], marker='x', color='r', zorder=10)
        else:
            plt.scatter(x1[i], x2[i], marker='.', color='b', zorder=10)
        #end if
    #end for
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show:
        plt.show()

def Prepare3DData(net, count):
    x = np.linspace(0,1,count)
    y = np.linspace(0,1,count)
    X, Y = np.meshgrid(x, y)
    if net is not None:
        input = np.hstack((X.ravel().reshape(count*count,1),Y.ravel().reshape(count*count,1)))
        net.inference(input)
    return X, Y

def DrawGrid(Z, count):
    for i in range(count):
        for j in range(count):
            plt.plot(Z[:,j,0],Z[:,j,1],'-',c='gray',lw=0.1)
            plt.plot(Z[i,:,0],Z[i,:,1],'-',c='gray',lw=0.1)
    #end for

def ShowSourceData(dr):
    ShowDataHelper(dr.XTrain[:,0], dr.XTrain[:,1], dr.YTrain[:,0], "Source Data", "x1", "x2", False, False)
    # grid
    count=20
    X,Y = Prepare3DData(None, count)
    for i in range(count):
        for j in range(count):
            plt.plot(X[i],Y[j],'-',c='gray',lw=0.1)
            plt.plot(Y[i],X[j],'-',c='gray',lw=0.1)
        #end for
    #end for
    plt.show()

def ShowResult2D(net, dr, epoch):
    ShowDataHelper(dr.XTrain[:,0], dr.XTrain[:,1], dr.YTrain[:,0], 
                   "Classifier Result, epoch=" + str(epoch), "x1", "x2", False, False)
    count = 50
    X,Y = Prepare3DData(net, count)
    Z = net.output.reshape(count,count)
    plt.contourf(X, Y, Z, cmap=plt.cm.Spectral, zorder=1)
    plt.show()

def ShowTransformation(net, dr, epoch):
    # draw z1
    net.inference(dr.XTrain)
    ShowDataHelper(net.Z1[:,0], net.Z1[:,1], dr.YTrain[:,0], 
                   "Layer 1 - Linear Transform, epoch=" + str(epoch), "x1", "x2", False, False)
    #grid
    count = 20
    X,Y = Prepare3DData(net, count)
    Z = net.Z1.reshape(count,count,2)
    DrawGrid(Z, count)
    plt.show()

    #draw a1
    net.inference(dr.XTrain)
    ShowDataHelper(net.A1[:,0], net.A1[:,1], dr.YTrain[:,0], 
                   "Layer 1 - Activation, epoch=" + str(epoch), "x1", "x2", False, False)
    #grid
    count = 20
    X,Y = Prepare3DData(net, count)
    Z = net.A1.reshape(count,count,2)
    DrawGrid(Z, count)
    plt.show()


def train(dataReader, max_epoch):
    n_input = dataReader.num_feature
    n_hidden = 2
    n_output = 1
    eta, batch_size = 0.1, 5
    eps = 0.01

    hp = HyperParameters_2_0(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.BinaryClassifier, InitialMethod.Xavier)
    net = NeuralNet_2_1(hp, "Arc_221_epoch")
    
    #net.LoadResult()
    net.train(dataReader, 5, True)
    #net.ShowTrainingTrace()
    
    ShowTransformation(net, dataReader, max_epoch)
    ShowResult2D(net, dataReader, max_epoch)


if __name__ == '__main__':
    dataReader = DataReader_2_0(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.Shuffle()
    dataReader.GenerateValidationSet()

    ShowSourceData(dataReader)
    plt.show()

    train(dataReader, 20)
    train(dataReader, 50)
    train(dataReader, 100)
    train(dataReader, 150)
    train(dataReader, 200)
    train(dataReader, 600)
