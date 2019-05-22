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

from Level0_OverFitNet import *

if __name__ == '__main__':

    dataReader = LoadData()
    num_feature = dataReader.num_feature
    num_example = dataReader.num_example
    num_input = num_feature
    num_hidden = 30
    num_output = 10
    max_epoch = 30
    batch_size = 10
    learning_rate = 0.1
    eps = 0.08

    params = CParameters(
        learning_rate, max_epoch, batch_size, eps,
        LossFunctionName.CrossEntropy3, 
        InitialMethod.Xavier, 
        OptimizerName.SGD,
        RegularMethod.L2, lambd=0.05)

    Net(dataReader, num_input, num_hidden, num_output, params)

