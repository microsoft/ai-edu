# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from matplotlib import pyplot as plt
import numpy as np
from Level3_Base import *
from ExtendedDataReader.MnistImageDataReader import *

def load_data():
    dataReader = MnistImageDataReader(mode="timestep")
    #dataReader.ReadLessData(10000)
    dataReader.ReadData()
    dataReader.NormalizeX()
    dataReader.NormalizeY(NetType.MultipleClassifier, base=0)
    dataReader.Shuffle()
    dataReader.GenerateValidationSet(k=12)
    return dataReader


if __name__=='__main__':
    net_type = NetType.MultipleClassifier
    output_type = OutputType.LastStep
    num_step = 28
    dataReader = load_data()
    eta = 0.005
    max_epoch = 100
    batch_size = 128
    num_input = dataReader.num_feature
    num_hidden = 32
    num_output = dataReader.num_category
    model = str.format("Level3_MNIST_{0}_{1}_{2}_{3}_{4}_{5}_{6}", max_epoch, batch_size, num_step, num_input, num_hidden, num_output, eta)
    hp = HyperParameters_4_3(
        eta, max_epoch, batch_size, 
        num_step, num_input, num_hidden, num_output,
        output_type, net_type)
    n = net(hp, model)

    n.train(dataReader, checkpoint=0.5)
    n.loss_trace.ShowLossHistory(hp.toString(), XCoordinate.Iteration)
    n.test(dataReader)
    n.load_parameters(ParameterType.Best)
    n.test(dataReader)
