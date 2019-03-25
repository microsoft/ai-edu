# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
from LossFunction import * 
from Activations import *
from Level1_TwoLayerNN import *
from DataReader import * 
from GDOptimizer import *
from WeightsBias import *

x_data_name = "X9_3.npy"
y_data_name = "Y9_3.npy"

if __name__ == '__main__':

    dataReader = DataReader(x_data_name, y_data_name)
    XData,YData = dataReader.ReadData()
    X = dataReader.NormalizeX()
    Y = dataReader.ToOneHot()
    
    n_input, n_output = dataReader.num_feature, dataReader.num_category
    n_hidden = 4
    eta, batch_size, max_epoch = 0.1, 10, 10000
    eps = 0.05

    params = CParameters(n_input, n_output, n_hidden,
                         eta, max_epoch, batch_size, eps, 
                         LossFunctionName.CrossEntropy3, 
                         InitialMethod.Xavier,
                         OptimizerName.O_SGD)

    loss_history = CLossHistory()
    net = CTwoLayerNet()

    net.ShowData(XData, YData)

    wbs = net.train(dataReader, params, loss_history)

    trace = loss_history.GetMinimalLossData()
    trace.print_info()
    loss_history.ShowLossHistory(params)

    wbs_min = WeightsBias(params)
    wbs_min.W1 = trace.dict_weights["W1"]
    wbs_min.W2 = trace.dict_weights["W2"]
    wbs_min.B1 = trace.dict_weights["B1"]
    wbs_min.B2 = trace.dict_weights["B2"]
    net.ShowAreaResult(X, wbs_min)
    net.ShowData(X, YData)
    