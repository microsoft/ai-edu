# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

import time
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet_4_2 import *
from ExtendedDataReader.MnistImageDataReader import *

def LoadData(num_output):
    mdr = MnistImageDataReader("image")
    mdr.ReadLessData(1000)
    #mdr.ReadData()
    mdr.NormalizeX()
    mdr.NormalizeY(NetType.MultipleClassifier, base=0)
    mdr.Shuffle()
    mdr.GenerateValidationSet(k=12)
    return mdr

def model():
    num_output = 10
    dataReader = LoadData(num_output)

    max_epoch = 1
    batch_size = 128
    learning_rate = 0.1
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier)

    net = NeuralNet_4_2(params, "mnist_conv")
    
    c1 = ConvLayer_GPU((1,28,28), (8,3,3), (1,1), params)
    net.add_layer(c1, "c1")
    r1 = ActivatorLayer(Relu())
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(c1.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p1, "p1") 
  
    c3 = ConvLayer_GPU(p1.output_shape, (16,3,3), (1,1), params)
    net.add_layer(c3, "c3")
    r3 = ActivatorLayer(Relu())
    net.add_layer(r3, "relu3")
    p2 = PoolingLayer(c3.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p2, "p2")  

    f1 = FcLayer_2_0(p2.output_size, 32, params)
    net.add_layer(f1, "f1")
    r5 = ActivatorLayer(Relu())
    net.add_layer(r5, "relu5")

    f2 = FcLayer_2_0(f1.output_size, 10, params)
    net.add_layer(f2, "f2")
    s1 = ClassificationLayer(Softmax())
    net.add_layer(s1, "s1")

    net.train(dataReader, checkpoint=0.01, need_test=True)
    net.ShowLossHistory(XCoordinate.Iteration)

if __name__ == '__main__':
    model()
