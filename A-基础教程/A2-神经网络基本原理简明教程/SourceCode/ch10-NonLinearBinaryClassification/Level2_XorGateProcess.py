# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from Level2_XorGateHow import *

def ShowProcess2D(net, dataReader, epoch):
    net.inference(dataReader.XTest)
    # show z1    
    ShowDataHelper(net.Z1[:,0], net.Z1[:,1], dataReader.YTest[:,0], "net.Z1, epoch="+str(epoch), "Z1[0]", "Z1[1]", show=True)
    # show a1
    ShowDataHelper(net.A1[:,0], net.A1[:,1], dataReader.YTest[:,0], "net.A1, epoch="+str(epoch), "A1[0]", "A1[1]", show=True)
    # show sigmoid
    ShowDataHelper(net.Z2, net.A2, dataReader.YTest[:,0], "Z2->A2, epoch="+str(epoch), "Z2", "A2", show=False)
    x = np.linspace(-6,6)
    a = Sigmoid().forward(x)
    plt.plot(x,a)
    plt.show()

def ShowResultContour(net, dr, title):
    ShowDataHelper(dr.XTrain[:,0], dr.XTrain[:,1], dr.YTrain[:,0], title, "x1", "x2", show=False)
    X,Y,Z = Prepare3DData(net, 50)
    plt.contourf(X, Y, Z, cmap=plt.cm.Spectral)
    plt.show()

def train(epoch, dataReader):
    n_input = dataReader.num_feature
    n_hidden = 2
    n_output = 1
    eta, batch_size, max_epoch = 0.1, 1, epoch
    eps = 0.005
    hp = HyperParameters_2_0(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.BinaryClassifier, InitialMethod.Xavier)
    net = NeuralNet_2_1(hp, "Xor_221_epoch")
    net.train(dataReader, 100, False)
    epoch = net.GetEpochNumber()
    ShowProcess2D(net, dataReader, epoch)
    ShowResultContour(net, dataReader, str.format("{0},epoch={1}", hp.toString(), epoch))

if __name__ == '__main__':
    dataReader = XOR_DataReader()
    dataReader.ReadData()

    train(500, dataReader)
    train(1500, dataReader)
    train(2000, dataReader)
    train(2500, dataReader)
    train(6000, dataReader)



    
