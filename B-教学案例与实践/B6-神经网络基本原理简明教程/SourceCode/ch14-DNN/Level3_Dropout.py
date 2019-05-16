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

from MnistImageDataReader import *

train_image_file = 'train-images-10'
train_label_file = 'train-labels-10'
test_image_file = 'test-images-10'
test_label_file = 'test-labels-10'

def LoadData():
    mdr = MnistImageDataReader(train_image_file, train_label_file, test_image_file, test_label_file, "vector")
    mdr.ReadLessData(1000)
    mdr.Normalize()
    mdr.GenerateDevSet()
    return mdr


def OverFitNet():
    dataReader = LoadData()
    num_feature = dataReader.num_feature
    num_example = dataReader.num_example
    num_input = num_feature
    num_hidden1 = 64
    num_hidden2 = 32
    num_output = 10
    max_epoch = 100
    batch_size = 100
    learning_rate = 0.2
    eps = 0.08

    params = CParameters(learning_rate, max_epoch, batch_size, eps,
                        LossFunctionName.CrossEntropy3, 
                        InitialMethod.Xavier, 
                        OptimizerName.SGD)

    net = NeuralNet(params)

    fc1 = FcLayer(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")

    sigmoid = ActivatorLayer(Relu())
    net.add_layer(sigmoid, "sigmoid")

    fc2 = FcLayer(num_hidden1, num_hidden2, params)
    net.add_layer(fc2, "fc2")

    tanh = ActivatorLayer(Relu())
    net.add_layer(tanh, "tanh")

    fc3 = FcLayer(num_hidden2, num_output, params)
    net.add_layer(fc3, "fc3")

    softmax = ActivatorLayer(Softmax())
    net.add_layer(softmax, "softmax")

    net.train(dataReader, checkpoint=1)
    
    net.ShowLossHistory(0, None, 0, 1)

def DropoutNet():
    dataReader = LoadData()
    num_feature = dataReader.num_feature
    num_example = dataReader.num_example
    num_input = num_feature
    num_hidden1 = 64
    num_hidden2 = 32
    num_output = 10
    max_epoch = 100
    batch_size = 100
    learning_rate = 0.2
    eps = 0.08

    params = CParameters(learning_rate, max_epoch, batch_size, eps,
                        LossFunctionName.CrossEntropy3, 
                        InitialMethod.Xavier, 
                        OptimizerName.SGD)

    net = NeuralNet(params)

    fc1 = FcLayer(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")

    sigmoid = ActivatorLayer(Relu())
    net.add_layer(sigmoid, "sigmoid")

    fc2 = FcLayer(num_hidden1, num_hidden2, params)
    net.add_layer(fc2, "fc2")

    tanh = ActivatorLayer(Relu())
    net.add_layer(tanh, "tanh")

    dp1 = DropoutLayer(num_hidden2, 0.5)
    net.add_layer(dp1, "dp1")

    fc3 = FcLayer(num_hidden2, num_output, params)
    net.add_layer(fc3, "fc3")

    dp2 = DropoutLayer(num_output, 0.5)
    net.add_layer(dp2, "dp2")


    softmax = ActivatorLayer(Softmax())
    net.add_layer(softmax, "softmax")

    net.train(dataReader, checkpoint=1)
    
    net.ShowLossHistory(0, None, 0, 1)


if __name__ == '__main__':
    # OverFitNet()
    DropoutNet()
