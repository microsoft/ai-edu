# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from HelperClass2.NeuralNet_2_2 import *
from HelperClass2.Visualizer_1_1 import *

train_data_name = "../../Data/ch11.train.npz"
test_data_name = "../../Data/ch11.test.npz"


def Show3D(net, dr):
    X,Y = dr.GetTestSet()
    net.inference(X)

    colors = ['b', 'r', 'g']
    shapes = ['o', 'x', 's']

    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig)
    count = Y.shape[0]
    for i in range(count):
        for j in range(Y.shape[1]):
            if Y[i,j] == 1:
                ax.scatter(net.Z1[i,0],net.Z1[i,1],net.Z1[i,2], color=colors[j], marker=shapes[j])
    plt.show()

    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig)
    count = Y.shape[0]
    for i in range(count):
        for j in range(Y.shape[1]):
            if Y[i,j] == 1:
                ax.scatter(net.A1[i,0],net.A1[i,1],net.A1[i,2], color=colors[j], marker=shapes[j])
    plt.show()


if __name__ == '__main__':
    dataReader = DataReader_2_0(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.NormalizeY(NetType.MultipleClassifier, base=1)

    dataReader.NormalizeX()
    dataReader.Shuffle()
    dataReader.GenerateValidationSet()

    n_input = dataReader.num_feature
    n_hidden = 3
    n_output = dataReader.num_category
    eta, batch_size, max_epoch = 0.1, 10, 5000
    eps = 0.1

    hp = HyperParameters_2_0(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.MultipleClassifier, InitialMethod.Xavier)
    net = NeuralNet_2_2(hp, "Bank_233_2")
    
    #net.LoadResult()
    net.train(dataReader, 100, True)
    net.ShowTrainingHistory()

    Show3D(net, dataReader)
