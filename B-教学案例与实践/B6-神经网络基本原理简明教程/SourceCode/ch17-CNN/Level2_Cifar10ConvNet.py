# Copyright (c) Microsoft.  All rights reserved.
# Licensed under the MIT license.  See LICENSE file in the project root for full license information.

import time

from MiniFramework.NeuralNet_4_2 import *
from ExtendedDataReader.CifarImageDataReader import *

def LoadData():
    print("reading data...")
    mdr = CifarImageDataReader("image")
    mdr.ReadData()
    mdr.NormalizeY(NetType.MultipleClassifier, base=0)
    mdr.Shuffle()
    mdr.GenerateValidationSet(k=20)
    return mdr

def model():
    num_output = 10
    dataReader = LoadData()

    max_epoch = 1
    batch_size = 32
    learning_rate = 0.01
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier)

    net = NeuralNet_4_2(params, "cifar_conv")
    
    c1 = ConvLayer((3,32,32), (32,5,5), (1,2), params)
    net.add_layer(c1, "c1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")
    p1 = PoolingLayer(c1.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p1, "p1") 

    c2 = ConvLayer(p1.output_shape, (32,5,5), (1,1), params)
    net.add_layer(c2, "c2")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "relu2")
    p2 = PoolingLayer(c2.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p2, "p2") 
  
    f1 = FcLayer_2_0(p2.output_size, 512, params)
    net.add_layer(f1, "f1")
    r3 = ActivationLayer(Relu())
    net.add_layer(r3, "relu3")

    f2 = FcLayer_2_0(f1.output_size, 128, params)
    net.add_layer(f2, "f2")
    r4 = ActivationLayer(Relu())
    net.add_layer(r4, "relu4")
    
    f3 = FcLayer_2_0(f2.output_size, 10, params)
    net.add_layer(f3, "f3")
    s1 = ClassificationLayer(Softmax())
    net.add_layer(s1, "s1")

    net.load_parameters()

    net.train(dataReader, checkpoint=0.05, need_test=True)
    net.ShowLossHistory(XCoordinate.Iteration)

if __name__ == '__main__':
    model()
