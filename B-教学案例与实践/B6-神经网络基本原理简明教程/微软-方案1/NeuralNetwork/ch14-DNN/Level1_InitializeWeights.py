# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet import *
from MiniFramework.GDOptimizer import *
from MiniFramework.LossFunction import *
from MiniFramework.Parameters import *
from MiniFramework.WeightsBias import *
from MiniFramework.Activators import *


def net():

    max_epoch = 1
    batch_size = 5
    learning_rate = 0.02
    eps = 0.01

    params = CParameters(learning_rate, max_epoch, batch_size, eps,
                        LossFunctionName.CrossEntropy3, 
                        InitialMethod.Normal, 
                        OptimizerName.SGD)

    loss_history = CLossHistory()

    net = NeuralNet(params)

    #num_hidden = [256,128,64,32,16]
    num_hidden = [100,100,100,100,100]
    count = len(num_hidden)
    layers = []

    for i in range(count-1):
        fc = FcLayer(num_hidden[i], num_hidden[i+1], Sigmoid())
        layers.append(fc)
        net.add_layer(fc)

    x = np.random.randn(1000, num_hidden[0]).T
    #x = np.random.normal(size=num_hidden[0]).T

    a = []
    a.append(x)

    for i in range(count):
        f = layers[0].forward(a[i])
        a.append(f)


    for i in range(count):
        plt.subplot(1, count, i+1)
        plt.title(str(i+1) + "-layer")
        plt.ylim(0,7000)
        plt.hist(a[i].flatten(), 30, range=(0,1))
    plt.show()

if __name__ == '__main__':
    net()
