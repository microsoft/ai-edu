# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from Level1_XorGateClassifier import *

def ShowProcess2D(net, dataReader, epoch):
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
    plt.title(str.format("net.Z1, epoch={0}", epoch))
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
    plt.title(str.format("net.A1, epoch={0}", epoch))
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
    plt.title(str.format("Logistic, epoch={0}", epoch))
    plt.xlabel("Z2")
    plt.ylabel("A2")
    plt.show()

def ShowResult3D(net, dr, title):
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

    plt.title(title)
    plt.show()

def train(epoch, dataReader):
    n_input = dataReader.num_feature
    n_hidden = 2
    n_output = 1
    eta, batch_size, max_epoch = 0.1, 1, epoch
    eps = 0.005
    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.BinaryClassifier, InitialMethod.Xavier)
    net = NeuralNet2(hp, "Xor_221_epoch")
    net.train(dataReader, 100, False)
    epoch = net.GetEpochNumber()
    ShowProcess2D(net, dataReader, epoch)
    ShowResult3D(net, dataReader, str.format("{0},epoch={1}", hp.toString(), epoch))

if __name__ == '__main__':
    dataReader = XOR_DataReader()
    dataReader.ReadData()
    dataReader.GenerateValidationSet()

    #train(500, dataReader)
    #train(1500, dataReader)
    train(2500, dataReader)
   # train(6000, dataReader)



    
