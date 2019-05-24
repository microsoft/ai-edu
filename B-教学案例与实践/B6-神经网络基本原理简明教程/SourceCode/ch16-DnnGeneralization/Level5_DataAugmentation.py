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

from MnistAugmentationReader import *

train_image_file = 'train-images-10'
train_label_file = 'train-labels-10'
test_image_file = 'test-images-10'
test_label_file = 'test-labels-10'

def LoadData():
    mdr = MnistAugmentationReader(None, None, test_image_file, test_label_file, "vector")
    mdr.ReadData()
    mdr.Normalize()
    mdr.Shuffle()
    mdr.GenerateDevSet(k=10)
    return mdr

#def Net(dataReader, num_input, num_hidden1, num_hidden2, num_hidden3, num_hidden4, num_output, params):
def Net(dataReader, num_input, num_hidden, num_output, params):
    net = NeuralNet(params)

    fc1 = FcLayer(num_input, num_hidden, params)
    net.add_layer(fc1, "fc1")
    relu1 = ActivatorLayer(Relu())
    net.add_layer(relu1, "relu1")
    
    fc2 = FcLayer(num_hidden, num_hidden, params)
    net.add_layer(fc2, "fc2")
    relu2 = ActivatorLayer(Relu())
    net.add_layer(relu2, "relu2")

    fc3 = FcLayer(num_hidden, num_hidden, params)
    net.add_layer(fc3, "fc3")
    relu3 = ActivatorLayer(Relu())
    net.add_layer(relu3, "relu3")

    fc4 = FcLayer(num_hidden, num_hidden, params)
    net.add_layer(fc4, "fc4")
    relu4 = ActivatorLayer(Relu())
    net.add_layer(relu4, "relu4")
    
    fc5 = FcLayer(num_hidden, num_output, params)
    net.add_layer(fc5, "fc5")
    softmax = ActivatorLayer(Softmax())
    net.add_layer(softmax, "softmax")

    net.train(dataReader, checkpoint=1, need_test=True)
    
    net.ShowLossHistory()


if __name__ == '__main__':

    dataReader = LoadData()
    num_feature = dataReader.num_feature
    num_example = dataReader.num_example
    num_input = num_feature
    num_hidden = 30
    num_output = 10
    max_epoch = 100
    batch_size = 100
    learning_rate = 0.1
    eps = 1e-2

    params = CParameters(
        learning_rate, max_epoch, batch_size, eps,                        
        LossFunctionName.CrossEntropy3, 
        InitialMethod.Xavier, 
        OptimizerName.SGD)

    #Net(dataReader, num_input, num_hidden, num_hidden, num_hidden, num_hidden, num_output, params)
    Net(dataReader, num_input, num_hidden, num_output, params)
