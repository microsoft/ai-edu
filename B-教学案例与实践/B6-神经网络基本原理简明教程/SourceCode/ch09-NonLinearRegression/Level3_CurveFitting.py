# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from HelperClass2.NeuralNet2 import *
from HelperClass2.DataReader import *

x_data_name = "../../Data/ch09.train.npz"
y_data_name = "../../Data/ch09.test.npz"

def ShowResult(net, dataReader, title):
    # draw train data
    X,Y = dataReader.XTrain, dataReader.YTrain
    plt.plot(X[:,0], Y[:,0], '.', c='b')
    # create and draw visualized validation data
    TX = np.linspace(0,1,100).reshape(100,1)
    TY = net.inference(TX)
    plt.plot(TX, TY, 'x', c='r')
    plt.title(title)
    plt.show()
#end def

def ShowResult3D(net, title):
    # draw train data
    # create and draw visualized validation data
    TX = np.linspace(0,1,100).reshape(100,1)
    TY = net.inference(TX)
    fig = plt.figure()
    ax = Axes3D(fig)
    #plt.plot(net.Z1[:,0],net.Z1[:,1],net.Z1[:,2],'.',c='black')
    plt.plot(net.A1[:,0],net.A1[:,1],net.A1[:,2],'.',c='black')
    plt.plot(net.A1[:,0],net.A1[:,1],net.Z2[:,0],'.',c='r')
    plt.plot(net.A1[:,0],net.A1[:,2],net.Z2[:,0],'.',c='g')
    plt.plot(net.A1[:,1],net.A1[:,2],net.Z2[:,0],'.',c='b')
    plt.show()

#end def


def ShowResult2D(net, title):
    # draw train data
    # create and draw visualized validation data
    TX = np.linspace(0,1,100).reshape(100,1)
    TY = net.inference(TX)
    fig = plt.figure()
    
    plt.plot(TX,np.zeros((100,1)),'x',c='cyan')
    plt.plot(TX,net.Z2[:,0],'.',c='black')
    #plt.plot(net.A1[:,0],net.A1[:,1],'.',c='cyan')
    #plt.plot(TX,net.Z1[:,0],'.',c='r')
    #plt.plot(TX,net.Z1[:,1],'.',c='g')
    #plt.plot(TX,net.Z1[:,2],'.',c='b')
    plt.plot(TX,net.A1[:,0],'.',c='r')
    plt.plot(TX,net.A1[:,1],'.',c='g')
    plt.plot(TX,net.A1[:,2],'.',c='b')
    #plt.plot(net.Z2[:,0],net.A1[:,0],'.',c='r')
    #plt.plot(net.A1[:,1],net.Z2[:,0],'.',c='g')
    plt.show()

if __name__ == '__main__':
    dataReader = DataReader(x_data_name, y_data_name)
    dataReader.ReadData()
    dataReader.GenerateValidationSet()

    n_input, n_hidden, n_output = 1, 3, 1
    eta, batch_size, max_epoch = 0.1, 10, 10000
    eps = 0.001

    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.Fitting, InitialMethod.Xavier)
    net = NeuralNet2(hp, "model_131")

    net.LoadResult()
    print(net.wb1.W)
    print(net.wb1.B)
    print(net.wb2.W)
    print(net.wb2.B)

    #net.train(dataReader, 50, True)
    #net.ShowTrainingTrace()
    #ShowResult(net, dataReader, hp.toString())
    #ShowResult3D(net, hp.toString())
    ShowResult2D(net, hp.toString())
