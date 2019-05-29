# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from MiniFramework.NeuralNet import *
from MiniFramework.Optimizer import *
from MiniFramework.LossFunction import *
from MiniFramework.Parameters import *
from MiniFramework.WeightsBias import *
from MiniFramework.ActivatorLayer import *
from MiniFramework.DropoutLayer import *

from MnistBaggingReader import *
from Level0_OverFitNet import *

train_image_file_temp = 'level6_{0}.npz'
test_image_file = 'test-images-10'
test_label_file = 'test-labels-10'

def LoadData(index):
    train_image_file = str.format(train_image_file_temp, index)
    mdr = MnistBaggingReader(train_image_file, None, test_image_file, test_label_file, "vector")
    mdr.ReadData()
    mdr.Normalize()
    mdr.Shuffle()
    mdr.GenerateDevSet(k=10)
    return mdr

def train(dataReader):
    num_feature = dataReader.num_feature
    num_example = dataReader.num_example
    num_input = num_feature
    num_hidden = 30
    num_output = 10
    max_epoch = 50
    batch_size = 10
    learning_rate = 0.1
    eps = 1e-2

    params = CParameters(
        learning_rate, max_epoch, batch_size, eps,                        
        LossFunctionName.CrossEntropy3, 
        InitialMethod.Xavier, 
        OptimizerName.SGD)

    net = Net(dataReader, num_input, num_hidden, num_output, params, show_history=False)
    return net

if __name__ == '__main__':
    nets = []
    net_count = 9
    for i in range(net_count):
        dataReader = LoadData(i)
        net = train(dataReader)
        nets.append(net)
    # test
    test_count = dataReader.num_test
#    test_count = 100
    dataReader = LoadData(0)
    predict_array = None
    for i in range(net_count):
        test_x, test_y = dataReader.GetBatchTestSamples(test_count, 0)
        output = nets[i].inference(test_x)
        predict = np.argmax(output, axis=0)
        if i == 0:
            predict_array = predict
        else:
            predict_array = np.vstack((predict_array, predict))
        # end if
    # end for

    ra = np.zeros(test_count)
    for i in range(test_count):
        ra[i] = np.argmax(np.bincount(predict_array[:,i]))

    ry = np.argmax(test_y, axis=0)
    r = (ra == ry)
    correct = r.sum()
    print(correct)