# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet import *
from MiniFramework.Optimizer import *
from MiniFramework.LossFunction import *
from MiniFramework.Parameters import *
from MiniFramework.WeightsBias import *
from MiniFramework.ActivatorLayer import *
from MiniFramework.DropoutLayer import *
from MiniFramework.DataReader import *

x_data_name = "X16.dat"
y_data_name = "Y16.dat"

def LoadData():
    dataReader = DataReader(x_data_name, y_data_name)
    dataReader.ReadData()
    dataReader.Normalize()
    return dataReader


def L2Net(num_input, num_hidden1, num_hidden2, num_output, params):
    net = NeuralNet(params)

    fc1 = FcLayer(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")

    relu1 = ActivatorLayer(Tanh())
    net.add_layer(relu1, "relu1")

    fc2 = FcLayer(num_hidden1, num_hidden2, params)
    net.add_layer(fc2, "fc2")

    relu2 = ActivatorLayer(Tanh())
    net.add_layer(relu2, "relu2")

    fc3 = FcLayer(num_hidden2, num_output, params)
    net.add_layer(fc3, "fc3")

    softmax = ActivatorLayer(Sigmoid())
    net.add_layer(softmax, "softmax")

    net.train(dataReader, checkpoint=1, need_test=False)
    
    net.ShowLossHistory()
    plot_decision_boundary(dataReader, net)


def plot_decision_boundary(dataReader, net):
    X = dataReader.X
    Y = dataReader.Y
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - .5, X[0, :].max() + .5
    y_min, y_max = X[1, :].min() - .5, X[1, :].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = net.inference(np.c_[xx.ravel(), yy.ravel()].T)#
    Z = np.reshape(Z, xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    
    for i in range(200):
        if Y[0,i] == 0:
            plt.plot(X[0,i], X[1,i], 'o', c='g')
        else:
            plt.plot(X[0,i], X[1,i], 'x', c='r')

    plt.show()


if __name__ == '__main__':
    dataReader = LoadData()
    num_feature = dataReader.num_feature
    num_example = dataReader.num_example
    num_input = num_feature
    num_hidden1 = 8
    num_hidden2 = 4
    num_output = 1
    max_epoch = 1000
    batch_size = 10
    learning_rate = 0.1
    eps = 0.01

    params = CParameters(learning_rate, max_epoch, batch_size, eps,
                        LossFunctionName.CrossEntropy2, InitialMethod.Xavier, OptimizerName.SGD
                        )

    L2Net(num_input, num_hidden1, num_hidden2, num_output, params)

