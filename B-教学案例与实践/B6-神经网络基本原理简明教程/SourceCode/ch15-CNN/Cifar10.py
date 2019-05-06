# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

# coding: utf-8

import pickle
import time
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet import *
from MiniFramework.GDOptimizer import *
from MiniFramework.LossFunction import *
from MiniFramework.Parameters import *
from MiniFramework.WeightsBias import *
from MiniFramework.Activators import *
from MiniFramework.ConvLayer import *
from MiniFramework.PoolingLayer import *


from CifarImageReader import *

file_1 = "..\\Data\\data_batch_1.bin"
file_2 = "..\\Data\\data_batch_2.bin"
file_3 = "..\\Data\\data_batch_3.bin"
file_4 = "..\\Data\\data_batch_4.bin"
file_5 = "..\\Data\\data_batch_5.bin"
test_file = "..\\Data\\test_batch.bin"

def ReadData():
    dr = CifarImageReader(file_1, file_2, file_3, file_4, file_5, test_file)
    dr.ReadData()
    dr.HoldOut(10)
    print(dr.num_validation, dr.num_example, dr.num_test, dr.num_train)
    return dr

def net():
    num_output = 10
    dr = ReadData()

    max_epoch = 1
    batch_size = 50
    eta = 0.001
    eps = 0.01
    params = CParameters(eta, max_epoch, batch_size, eps,
                    LossFunctionName.CrossEntropy3, 
                    InitialMethod.Xavier, 
                    OptimizerName.Adam)

    loss_history = CLossHistory()

    net = NeuralNet(params)

    c1 = ConvLayer((3,32,32), (32,3,3), (1,1), Relu(), params)
    net.add_layer(c1, "c1")

    p1 = PoolingLayer(c1.output_shape, (2,2,), 2, PoolingTypes.MAX)
    net.add_layer(p1, "p1")

    c2 = ConvLayer(p1.output_shape, (64,3,3), (1,1), Relu(), params)
    net.add_layer(c2, "c2")

    p2 = PoolingLayer(c2.output_shape, (2,2,), 2, PoolingTypes.MAX)
    net.add_layer(p2, "p2")

    f1 = FcLayer(p2.output_size, 512, Relu(), params)
    net.add_layer(f1, "f1")

    f2 = FcLayer(f1.output_size, 10, Softmax(), params)
    net.add_layer(f2, "f2")

    net.train(dr, loss_history)

    loss_history.ShowLossHistory(params)


if __name__ == '__main__':
    net()
