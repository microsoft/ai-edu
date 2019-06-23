# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from Level1_XorGateClassifier import *
    
def ShowSourceData(dataReader):
    fig = plt.figure(figsize=(6,6))

    X0 = dataReader.GetSetByLabel("train", 0)
    plt.scatter(X0[:,0], X0[:,1], marker='^', color='r', s=200)

    X1 = dataReader.GetSetByLabel("train", 1)
    plt.scatter(X1[:,0], X1[:,1], marker='o', color='b', s=200)

    plt.grid()
    plt.title("XOR source data")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

def ShowProcess2D(net, dataReader):
    net.inference(dataReader.XTrain)
    # show z1    
    fig = plt.figure(figsize=(6,6))
    for i in range(dataReader.num_train):
        if dataReader.YTrain[i,0] == 0:
            plt.scatter(net.Z1[i,0],net.Z1[i,1],marker='^',color='r',s=200)
        else:
            plt.scatter(net.Z1[i,0],net.Z1[i,1],marker='o',color='b',s=200)
        #end if
    #end for
    plt.grid()
    plt.title("net.Z1")
    plt.xlabel("Z(1,1)")
    plt.ylabel("Z(1,2)")
    plt.show()
    # show a1
    fig = plt.figure(figsize=(6,6))
    for i in range(dataReader.num_train):
        if dataReader.YTrain[i,0] == 0:
            plt.scatter(net.A1[i,0],net.A1[i,1],marker='^',color='r',s=200)
        else:
            plt.scatter(net.A1[i,0],net.A1[i,1],marker='o',color='b',s=200)
        #end if
    #end for
    plt.grid()
    plt.xlabel("A(1,1)")
    plt.ylabel("A(1,2)")
    plt.title("net.A1")
    plt.show()
    # show sigmoid
    fig = plt.figure(figsize=(6,6))
    x = np.linspace(-6,6)
    a = Sigmoid().forward(x)
    plt.plot(x,a)

    for i in range(dataReader.num_train):
        if dataReader.YTrain[i,0] == 0:
            plt.scatter(net.Z2[i,0],net.A2[i,0],marker='^',color='r',s=200)
        else:
            plt.scatter(net.Z2[i,0],net.A2[i,0],marker='o',color='b',s=200)
    plt.grid()
    plt.title("Result")
    plt.xlabel("Z2")
    plt.ylabel("A2")
    plt.show()

def ShowResult2D(net, title):
    fig = plt.figure(figsize=(6,6))
    
    X0 = dataReader.GetSetByLabel("train", 0)
    plt.scatter(X0[:,0], X0[:,1], marker='^', color='r', s=200, zorder=10)
    X1 = dataReader.GetSetByLabel("train", 1)
    plt.scatter(X1[:,0], X1[:,1], marker='o', color='b', s=200, zorder=10)

    count = 50
    x1 = np.linspace(0,1,count)
    x2 = np.linspace(0,1,count)
    for i in range(count):
        for j in range(count):
            x = np.array([x1[i],x2[j]]).reshape(1,2)
            output = net.inference(x)
            if output[0,0] >= 0.5:
                plt.plot(x[0,0], x[0,1], 's', c='m', zorder=1)
            else:
                plt.plot(x[0,0], x[0,1], 's', c='y', zorder=1)
            # end if
        # end for
    # end for
    plt.title(title)

    plt.show()
    
#end def

def ShowResult3D(net, dr):
    fig = plt.figure(figsize=(6,6))

    count = 50
    x = np.linspace(0,1,count)
    y = np.linspace(0,1,count)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((count, count))

    input = np.hstack((X.ravel().reshape(count*count,1),Y.ravel().reshape(count*count,1)))
    output = net.inference(input)
    Z = output.reshape(count,count)
    plt.contourf(X, Y, Z, cmap=plt.cm.Spectral)

    X0 = dataReader.GetSetByLabel("train", 0)
    plt.scatter(X0[:,0], X0[:,1], marker='^', color='r', s=200)

    X1 = dataReader.GetSetByLabel("train", 1)
    plt.scatter(X1[:,0], X1[:,1], marker='o', color='b', s=200)

    plt.show()

if __name__ == '__main__':
    dataReader = XOR_DataReader()
    dataReader.ReadData()
    dataReader.GenerateValidationSet()

    ShowSourceData(dataReader)

    n_input = dataReader.num_feature
    n_hidden = 2
    n_output = 1
    eta, batch_size, max_epoch = 0.1, 1, 10000
    eps = 0.005

    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.BinaryClassifier, InitialMethod.Xavier)
    net = NeuralNet2(hp, "Xor_221")

    #net.train(dataReader, 100, True)
    #net.ShowTrainingTrace()
    net.LoadResult()

    ShowProcess2D(net, dataReader)
    ShowResult2D(net, hp.toString())    
    ShowResult3D(net, dataReader)
