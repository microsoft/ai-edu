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
    max_epoch = 10
    batch_size = 32
    learning_rate = 0.1
    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.MSRA,
        optimizer_name=OptimizerName.Momentum)

    net = NeuralNet_4_2(params, "cifar_conv")
    
    c1 = ConvLayer((3,32,32), (32,3,3), (1,1), params)
    net.add_layer(c1, "c1")
    r1 = ActivationLayer(Relu())
    net.add_layer(r1, "relu1")
    
    c2 = ConvLayer(c1.output_shape, (32,3,3), (1,0), params)
    net.add_layer(c2, "c2")
    r2 = ActivationLayer(Relu())
    net.add_layer(r2, "relu2")
    p2 = PoolingLayer(c2.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p2, "p2") 

    c3 = ConvLayer(p2.output_shape, (64,3,3), (1,1), params)
    net.add_layer(c3, "c3")
    r3 = ActivationLayer(Relu())
    net.add_layer(r3, "relu3")
    
    c4 = ConvLayer(c3.output_shape, (64,3,3), (1,0), params)
    net.add_layer(c4, "c4")
    r4 = ActivationLayer(Relu())
    net.add_layer(r4, "relu4")
    p4 = PoolingLayer(c4.output_shape, (2,2), 2, PoolingTypes.MAX)
    net.add_layer(p4, "p4") 
  
    f5 = FcLayer_2_0(p4.output_size, 512, params)
    net.add_layer(f5, "f5")
    r5 = ActivationLayer(Relu())
    net.add_layer(r5, "relu5")
    
    f6 = FcLayer_2_0(f5.output_size, num_output, params)
    net.add_layer(f6, "f6")
    s7 = ClassificationLayer(Softmax())
    net.add_layer(s7, "s7")

    return net

if __name__ == '__main__':
    dataReader = LoadData()
    net = model()
    net.train(dataReader, checkpoint=0.05, need_test=True)
    net.ShowLossHistory(XCoordinate.Iteration)
