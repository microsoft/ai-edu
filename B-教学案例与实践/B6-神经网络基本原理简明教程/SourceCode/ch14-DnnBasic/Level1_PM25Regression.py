# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from MiniFramework.NeuralNet import *
from MiniFramework.Optimizer import *
from MiniFramework.LossFunction import *
from MiniFramework.Parameters import *
from MiniFramework.WeightsBias import *
from MiniFramework.ActivatorLayer import *
from MiniFramework.DataReader import *

import numpy as np

train_file = "../../Data/PM25_Train.npz"
test_file = "../../Data/PM25_Test.npz"

def LoadData():
    dr = DataReader(train_file, test_file)
    dr.ReadData()
    dr.NormalizeX()
    dr.NormalizeY(YNormalizationMethod.Regression)
    dr.Shuffle()
    dr.GenerateValidationSet()
    return dr

if __name__ == '__main__':
    dr = LoadData()
    
    num_input = dr.num_feature
    num_hidden1 = 16
    num_hidden2 = 4
    num_output = 1

    max_epoch = 1000
    batch_size = 100
    learning_rate = 0.1
    eps = 0.001

    params = CParameters(
        learning_rate, max_epoch, batch_size, eps,
        LossFunctionName.MSE, 
        InitialMethod.MSRA, 
        OptimizerName.Momentum)

    net = NeuralNet(params, "PM25")

    fc1 = FcLayer(num_input, num_hidden1, params)
    net.add_layer(fc1, "fc1")
    sigmoid1 = ActivatorLayer(Relu())
    net.add_layer(sigmoid1, "sigmoid1")
    
    fc2 = FcLayer(num_hidden1, num_hidden2, params)
    net.add_layer(fc2, "fc2")
    sigmoid2 = ActivatorLayer(Relu())
    net.add_layer(sigmoid2, "sigmoid2")
    
    fc3 = FcLayer(num_hidden2, num_output, params)
    net.add_layer(fc3, "fc3")

    net.train(dr, checkpoint=10, need_test=True)
    net.ShowLossHistory()