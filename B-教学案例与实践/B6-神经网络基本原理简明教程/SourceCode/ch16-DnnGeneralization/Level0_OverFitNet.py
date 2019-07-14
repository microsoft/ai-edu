# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.pardir)

from MiniFramework.NeuralNet import *
from MiniFramework.Optimizer import *
from MiniFramework.LossFunction import *
from MiniFramework.HyperParameters3 import *
from MiniFramework.DataReader import *
from MiniFramework.WeightsBias import *
from MiniFramework.ActivatorLayer import *
from MiniFramework.DropoutLayer import *

train_data_name = "../../Data/ch16.train.npz"
test_data_name = "../../Data/ch16.test.npz"

def LoadData():
    dr = DataReader(train_data_name, test_data_name)
    dr.ReadData()
    dr.GenerateValidationSet(k=10)
    return dr

def Net(dataReader, params, num_hidden, show_history=True):
    net = NeuralNet(params, "aaa")

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
    softmax = ActivatorLayer(Logistic())
    net.add_layer(softmax, "logistic")

    net.train(dataReader, checkpoint=1, need_test=True)
    if show_history:
        net.ShowLossHistory()
    
    return net


if __name__ == '__main__':

    dataReader = LoadData()
    num_feature = 1
    num_input = num_feature
    num_hidden = 30
    num_output = 1
    max_epoch = 100
    batch_size = 5
    learning_rate = 0.1
    eps = 0.08

    hp = HyperParameters2(num_input, num_output, learning_rate, max_epoch, batch_size, eps, NetType.Fitting, InitialMethod.Xavier)
    Net(dataReader, hp, num_hidden, "aaa")
