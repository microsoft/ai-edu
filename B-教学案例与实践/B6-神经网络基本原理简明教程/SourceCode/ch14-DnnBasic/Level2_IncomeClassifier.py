# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math

from MiniFramework.NeuralNet import *
from MiniFramework.Optimizer import *
from MiniFramework.LossFunction import *
from MiniFramework.Parameters import *
from MiniFramework.WeightsBias import *
from MiniFramework.ActivatorLayer import *
from MiniFramework.DataReader import *

train_file = "../../Data/Income_Train.npz"
test_file = "../../Data/Income_Test.npz"

def LoadData():
    dr = DataReader(train_file, test_file)
    dr.ReadData()
    dr.NormalizeX()
    #dr.NormalizeY(YNormalizationMethod.BinaryClassifier)
    dr.Shuffle()
    dr.GenerateValidationSet()
    return dr


if __name__ == '__main__':
    dr = LoadData()
    
    num_input = dr.num_feature
    num_hidden1 = 32
    num_hidden2 = 8
    num_output = 1

    max_epoch = 100
    batch_size = 16
    learning_rate = 0.1
    eps = 0.001

    params = CParameters(
        learning_rate, max_epoch, batch_size, eps,
        LossFunctionName.CrossEntropy2,
        InitialMethod.Xavier, 
        OptimizerName.SGD)

    net = NeuralNet(params, "Income")

    fc1 = FcLayer(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    a1 = ActivatorLayer(Relu())
    net.add_layer(a1, "relu1")
    
    fc2 = FcLayer(num_hidden1, num_hidden2, params)
    net.add_layer(fc2, "fc2")
    a2 = ActivatorLayer(Relu())
    net.add_layer(a2, "relu2")

    fc3 = FcLayer(num_hidden2, num_output, params)
    net.add_layer(fc3, "fc3")
    sigmoid3 = ClassificationLayer(Sigmoid())
    net.add_layer(sigmoid3, "sigmoid3")

    net.load_parameters()

    net.train(dr, checkpoint=1, need_test=True)
    net.ShowLossHistory()
