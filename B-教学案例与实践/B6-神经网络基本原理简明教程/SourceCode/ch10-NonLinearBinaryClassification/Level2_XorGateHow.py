# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Level1_XorGateClassifier import *
    
def ShowSourceData(dataReader):
    DrawSamplePoints(dataReader.XTrain[:,0],dataReader.XTrain[:,1],dataReader.YTrain, "XOR Source Data", "x1", "x2")

def ShowProcess2D(net, dataReader):
    net.inference(dataReader.XTest)
    # show z1    
    DrawSamplePoints(net.Z1[:,0], net.Z1[:,1], dataReader.YTest, "net.Z1", "Z1[0]", "Z1[1]")
    # show a1
    DrawSamplePoints(net.A1[:,0], net.A1[:,1], dataReader.YTest, "net.A1", "A1[0]", "A1[1]")
    # show sigmoid
    DrawSamplePoints(net.Z2, net.A2, dataReader.YTrain, "Z2->A2", "Z2", "A2", show=False)
    x = np.linspace(-6,6)
    a = Sigmoid().forward(x)
    plt.plot(x,a)
    plt.show()

def DrawSamplePoints(x1, x2, y, title, xlabel, ylabel, show=True):
    assert(x1.shape[0] == x2.shape[0])
    fig = plt.figure(figsize=(6,6))
    count = x1.shape[0]
    for i in range(count):
        if y[i,0] == 0:
            plt.scatter(x1[i], x2[i], marker='^', color='r', s=200, zorder=10)
        else:
            plt.scatter(x1[i], x2[i], marker='o', color='b', s=200, zorder=10)
        #end if
    #end for
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show:
        plt.show()


def ShowResult2D(net, dr, title):
    print("please wait for a while...")
    DrawSamplePoints(dr.XTest[:,0], dr.XTest[:,1], dr.YTest, title, "x1", "x2", show=False)
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

def Prepare3DData(net, count):
    x = np.linspace(0,1,count)
    y = np.linspace(0,1,count)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((count, count))
    input = np.hstack((X.ravel().reshape(count*count,1),Y.ravel().reshape(count*count,1)))
    output = net.inference(input)
    Z = output.reshape(count,count)
    return X,Y,Z

def ShowResultContour(net, dr):
    DrawSamplePoints(dr.XTrain[:,0], dr.XTrain[:,1], dr.YTrain, "classification result", "x1", "x2", show=False)
    X,Y,Z = Prepare3DData(net, 50)
    plt.contourf(X, Y, Z, cmap=plt.cm.Spectral)
    plt.show()

def ShowResult3D(net, dr):
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig)
    X,Y,Z = Prepare3DData(net, 50)
    ax.plot_surface(X,Y,Z,cmap='rainbow')
    ax.set_zlim(0,1)
    # draw sample data in 3D space
    for i in range(dr.num_train):
        if dataReader.YTrain[i,0] == 0:
            ax.scatter(dataReader.XTrain[i,0],dataReader.XTrain[i,1],dataReader.YTrain[i,0],marker='^',c='r',s=200)
        else:
            ax.scatter(dataReader.XTrain[i,0],dataReader.XTrain[i,1],dataReader.YTrain[i,0],marker='o',c='b',s=200)

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
    ShowResult2D(net, dataReader, hp.toString())
    ShowResult3D(net, dataReader)
    ShowResultContour(net, dataReader)
