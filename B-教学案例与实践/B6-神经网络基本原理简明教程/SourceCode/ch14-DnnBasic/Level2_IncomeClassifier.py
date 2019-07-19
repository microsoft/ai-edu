# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math

from MiniFramework.NeuralNet_4_0 import *
from MiniFramework.ActivationLayer import *
from MiniFramework.ClassificationLayer import *
from MiniFramework.DataReader_2_0 import *

train_file = "../../Data/ch14.Income.train.npz"
test_file = "../../Data/ch14.Income.test.npz"

def LoadData():
    dr = DataReader_2_0(train_file, test_file)
    dr.ReadData()
    dr.NormalizeX()
    #dr.NormalizeY(YNormalizationMethod.BinaryClassifier)
    dr.Shuffle()
    dr.GenerateValidationSet()
    return dr


def model():
    dr = LoadData()
    
    num_input = dr.num_feature
    num_hidden1 = 64
    num_hidden2 = 64
    num_hidden3 = 32
    num_hidden4 = 16
    num_output = 1

    max_epoch = 100
    batch_size = 16
    learning_rate = 0.1
    eps = 1e-3

    params = HyperParameters_4_0(
        learning_rate, max_epoch, batch_size, eps,
        net_type=NetType.BinaryClassifier,
        init_method=InitialMethod.Xavier)

    net = NeuralNet_4_0(params, "Income")

    fc1 = FcLayer_1_0(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    a1 = ActivationLayer(Relu())
    net.add_layer(a1, "relu1")
    
    fc2 = FcLayer_1_0(num_hidden1, num_hidden2, params)
    net.add_layer(fc2, "fc2")
    a2 = ActivationLayer(Relu())
    net.add_layer(a2, "relu2")

    fc3 = FcLayer_1_0(num_hidden2, num_hidden3, params)
    net.add_layer(fc3, "fc3")
    a3 = ActivationLayer(Relu())
    net.add_layer(a3, "relu3")

    fc4 = FcLayer_1_0(num_hidden3, num_hidden4, params)
    net.add_layer(fc4, "fc4")
    a4 = ActivationLayer(Relu())
    net.add_layer(a4, "relu4")

    fc5 = FcLayer_1_0(num_hidden4, num_output, params)
    net.add_layer(fc5, "fc5")
    logistic = ClassificationLayer(Logistic())
    net.add_layer(logistic, "logistic")

    #net.load_parameters()

    net.train(dr, checkpoint=10, need_test=True)
    net.ShowLossHistory()


if __name__ == '__main__':
    model()