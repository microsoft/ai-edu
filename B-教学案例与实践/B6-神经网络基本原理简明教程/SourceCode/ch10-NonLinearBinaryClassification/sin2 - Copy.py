# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

from HelperClass2.DataReader import *
from HelperClass2.HyperParameters2 import *
from HelperClass2.NeuralNet2 import *

train_data_name = "../../Data/ch10.train.npz"
test_data_name = "../../Data/ch10.test.npz"

def ShowData(dr):
    fig = plt.figure(figsize=(6,6))

    X0 = dr.GetSetByLabel("train", 0)
    X1 = dr.GetSetByLabel("train", 1)
    plt.scatter(X0[:,0], X0[:,1], marker='x', color='r')
    plt.scatter(X1[:,0], X1[:,1], marker='.', color='b')

    # grid
    count=20
    x = np.linspace(0,1,count)
    y = np.linspace(0,1,count)
    X, Y = np.meshgrid(x, y)
    assert(X.shape==(count,count))
    for i in range(count):
        for j in range(count):
            plt.plot(X[i],Y[j],'-',c='gray',lw=0.1)
            plt.plot(Y[i],X[j],'-',c='gray',lw=0.1)

    plt.show()

def ShowResult3D(net, dr):
    #fig = plt.figure()
    #ax = Axes3D(fig)
    count = 50
    x = np.linspace(0,1,count)
    y = np.linspace(0,1,count)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((len(x), len(y)))

    input = np.hstack((X.ravel().reshape(2500,1),Y.ravel().reshape(2500,1)))
    output = net.inference(input)
    Z = output.reshape(len(x),len(y))
    plt.contourf(X, Y, Z, cmap=plt.cm.Spectral)
    #ax.plot_surface(X,Y,Z,cmap='rainbow')
    #ax.set_zlim(0,1.2)

    for i in range(dr.num_train):
        if dr.YTrain[i,0] == 0:
            #ax.scatter(dr.XTrain[i,0], dr.XTrain[i,1], dr.YTrain[i,0]+0.1, marker='x', color='r')
            plt.scatter(dr.XTrain[i,0], dr.XTrain[i,1], marker='x', color='r')
        else:
            #ax.scatter(dr.XTrain[i,0], dr.XTrain[i,1], dr.YTrain[i,0]+0.1, marker='.', color='b')
            plt.scatter(dr.XTrain[i,0], dr.XTrain[i,1], marker='.', color='b')

    plt.show()


def train(net, dataReader):
    ShowData(dataReader)
    plt.show()

    net.train(dataReader, 100, True)
    net.ShowTrainingTrace()
    
    ShowResult3D(net, dataReader)

def Show2D(net, dataReader):
    net.LoadResult()
    fig = plt.figure(figsize=(6,6))

    X0 = dataReader.GetSetByLabel("train", 0)
    X1 = dataReader.GetSetByLabel("train", 1)
    net.inference(X0)
    plt.scatter(net.A1[:,0], net.A1[:,1], marker='x', color='r')
    net.inference(X1)
    plt.scatter(net.A1[:,0], net.A1[:,1], marker='.', color='b')

    #grid
    count=20
    x = np.linspace(0,1,count)
    y = np.linspace(0,1,count)
    X, Y = np.meshgrid(x, y)
    input = np.hstack((X.ravel().reshape(count*count,1),Y.ravel().reshape(count*count,1)))
    net.inference(input)
    
    Z = net.A1.reshape(count,count,2)
    for i in range(count):
        for j in range(count):
            plt.plot(Z[:,j,0],Z[:,j,1],'-',c='gray',lw=0.1)
            plt.plot(Z[i,:,0],Z[i,:,1],'-',c='gray',lw=0.1)

    plt.show()

if __name__ == '__main__':
    dataReader = DataReader(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.Shuffle()
    dataReader.GenerateValidationSet()

    n_input = dataReader.num_feature
    n_hidden = 2
    n_output = 1
    eta, batch_size, max_epoch = 0.1, 5, 10000
    eps = 0.1

    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.BinaryClassifier, InitialMethod.Xavier)
    net = NeuralNet2(hp, "221")

    #train(net, dataReader)
    ShowData(dataReader)
    Show2D(net, dataReader)

