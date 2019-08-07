# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet_4_2 import *
from MiniFramework.ActivatorLayer import *

from ExtendedDataReader.MnistBaggingReader import *
from Level1_OverfittingNet_Classification import *

def LoadData(index):
    mdr = MnistBaggingReader("vector")
    mdr.ReadData(index)
    mdr.NormalizeX()
    mdr.NormalizeY(NetType.MultipleClassifier, base=0)
    mdr.Shuffle()
    mdr.GenerateValidationSet(k=10)
    return mdr

def train(dataReader):
    num_feature = dataReader.num_feature
    num_example = dataReader.num_example
    num_input = num_feature
    num_hidden = 30
    num_output = 10
    max_epoch = 50
    batch_size = 32
    learning_rate = 0.1

    params = HyperParameters_4_2(
        learning_rate, max_epoch, batch_size,
        net_type=NetType.MultipleClassifier,
        init_method=InitialMethod.Xavier)

    net = Net("ensemble", dataReader, num_input, num_hidden, num_output, params, show_history=False)
    return net

if __name__ == '__main__':
    nets = []
    net_count = 9
    for i in range(net_count):
        dataReader = LoadData(i)
        net = train(dataReader)
        nets.append(net)
    #end for
    # test
    test_count = dataReader.num_test
    dataReader = LoadData(0)
    predict_array = None
    for i in range(net_count):
        test_x, test_y = dataReader.GetBatchTestSamples(test_count, 0)
        output = nets[i].inference(test_x)
        predict = np.argmax(output, axis=1)
        if i == 0:
            predict_array = predict
        else:
            predict_array = np.vstack((predict_array, predict))
        # end if
    # end for

    # vote
    ra = np.zeros(test_count)
    for i in range(test_count):
        ra[i] = np.argmax(np.bincount(predict_array[:,i]))

    ry = np.argmax(test_y, axis=1)
    r = (ra == ry)
    correct = r.sum()
    print(correct)