# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Level1_LogicXorGate import *


def ShowResult3D(net, dataReader):
    fig = plt.figure()
    ax = Axes3D(fig)
    count = 50
    x = np.linspace(0,1,count)
    y = np.linspace(0,1,count)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((len(x), len(y)))

    for i in range(count):
        for j in range(count):
            input = np.array([x[i],y[j]]).reshape(1,2)
            output = net.inference(input)
            Z[i,j] = output[0,0]
            # end if
        # end for
    # end for
    ax.plot_surface(X,Y,Z,cmap='rainbow')
    ax.set_zlim(0,1)

    for i in range(dataReader.num_train):
        if dataReader.YTrain[i,0] == 0:
            ax.scatter(dataReader.XTrain[i,0],dataReader.XTrain[i,1],dataReader.YTrain[i,0],marker='^',c='r')
        else:
            ax.scatter(dataReader.XTrain[i,0],dataReader.XTrain[i,1],dataReader.YTrain[i,0],marker='o',c='g')

    plt.show()


def ShowProcess3D(net, dataReader):
    net.inference(dataReader.XTrain)

    for i in range(dataReader.num_train):
        if dataReader.YTrain[i,0] == 0:
            plt.plot(dataReader.XTrain[i,0],dataReader.XTrain[i,1],marker='^',c='r')
        else:
            plt.plot(dataReader.XTrain[i,0],dataReader.XTrain[i,1],marker='o',c='g')
    plt.grid()
    plt.title("X1:X2")
    plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(dataReader.num_train):
        if dataReader.YTrain[i,0] == 0:
            ax.scatter(net.Z1[i,0],net.Z1[i,1],net.Z1[i,2],marker='^',c='r')
        else:
            ax.scatter(net.Z1[i,0],net.Z1[i,1],net.Z1[i,2],marker='o',c='g')
    plt.grid()
    plt.title("net.Z1")
    plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(dataReader.num_train):
        if dataReader.YTrain[i,0] == 0:
            ax.scatter(net.A1[i,0],net.A1[i,1],net.A1[i,2],marker='^',c='r')
        else:
            ax.scatter(net.A1[i,0],net.A1[i,1],net.A1[i,2],marker='o',c='g')
    plt.grid()
    plt.title("net.A1")
    plt.show()

    x = np.linspace(-6,6)
    a = Sigmoid().forward(x)
    plt.plot(x,a)

    for i in range(dataReader.num_train):
        if dataReader.YTrain[i,0] == 0:
            plt.plot(net.Z2[i,0],net.A2[i,0],'^',c='r')
        else:
            plt.plot(net.Z2[i,0],net.A2[i,0],'o',c='g')
    plt.grid()
    plt.title("Z2:A2")
    plt.show()


if __name__ == '__main__':
    dataReader = XOR_DataReader()
    dataReader.ReadData()
    dataReader.GenerateValidationSet()

    n_input = dataReader.num_feature
    n_hidden = 3
    n_output = 1
    eta, batch_size, max_epoch = 0.1, 1, 10000
    eps = 0.005

    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.BinaryClassifier, InitialMethod.Xavier)
    net = NeuralNet2(hp, "xor_231")

    net.train(dataReader, 100, True)
    net.ShowTrainingTrace()

    print(Test(dataReader, net))

    ShowProcess3D(net, dataReader)
    ShowResult3D(net, dataReader)