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
from MnistAugmentationReader import *

from Level0_OverFitNet import *


train_image_file = 'train-images-10'
train_label_file = 'train-labels-10'
test_image_file = 'test-images-10'
test_label_file = 'test-labels-10'

def LoadData():
    mdr = MnistAugmentationReader(None, None, test_image_file, test_label_file, "vector")
    mdr.ReadData()
    mdr.Normalize()
    mdr.Shuffle()
    mdr.GenerateDevSet(k=10)
    return mdr

if __name__ == '__main__':

    dataReader = LoadData()
    num_feature = dataReader.num_feature
    num_example = dataReader.num_example
    num_input = num_feature
    num_hidden = 30
    num_output = 10
    max_epoch = 200
    batch_size = 100
    learning_rate = 0.1
    eps = 1e-2

    params = CParameters(
        learning_rate, max_epoch, batch_size, eps,                        
        LossFunctionName.CrossEntropy3, 
        InitialMethod.Xavier, 
        OptimizerName.SGD)

    Net(dataReader, num_input, num_hidden, num_output, params)
