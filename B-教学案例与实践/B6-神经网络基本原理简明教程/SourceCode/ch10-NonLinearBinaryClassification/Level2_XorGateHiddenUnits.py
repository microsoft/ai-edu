# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from Level1_XorGateClassifier import *
    
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

def train(hidden, dataReader):
    n_input = dataReader.num_feature
    n_hidden = hidden
    n_output = 1
    eta, batch_size, max_epoch = 0.1, 1, 10000
    eps = 0.005
    hp = HyperParameters2(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.BinaryClassifier, InitialMethod.Xavier)
    net = NeuralNet2(hp, "Xor_2N1")
    net.train(dataReader, 100, False)
    epoch = net.GetEpochNumber()
    ShowResult3D(net, dataReader, str.format("{0},epoch={1}", hp.toString(), epoch))

if __name__ == '__main__':
    dataReader = XOR_DataReader()
    dataReader.ReadData()
    dataReader.GenerateValidationSet()

    train(1, dataReader)
    train(2, dataReader)
    train(3, dataReader)
    train(4, dataReader)
    train(8, dataReader)
    train(16, dataReader)



    
