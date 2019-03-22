# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
from LossFunction import * 
from Parameters import *
from Activations import *
from Level1_TwoLayer import *
from DataOperator import * 

x_data_name = "X9_3.npy"
y_data_name = "Y9_3.npy"

if __name__ == '__main__':

    XData,YData = DataOperator.ReadData(x_data_name, y_data_name)
    norm = DataOperator("min_max")
    X = norm.NormalizeData(XData)
    num_category = 3
    Y = DataOperator.ToOneHot(YData, num_category)

    num_example = X.shape[1]
    num_feature = X.shape[0]
    
    n_input, n_hidden, n_output = num_feature, 4, num_category
    eta, batch_size, max_epoch = 0.01, 10, 10000
    eps = 0.05
    init_method = InitialMethod.xavier

    params = CParameters(num_example, n_input, n_output, n_hidden, eta, max_epoch, batch_size, LossFunctionName.CrossEntropy3, eps, init_method)

    loss_history = CLossHistory()
    net = CTwoLayerNet()

    net.ShowData(XData, YData)

    dict_weights = net.train(X, Y, params, loss_history)

    bookmark = loss_history.GetMinimalLossData()
    bookmark.print_info()
    loss_history.ShowLossHistory(params)

    net.ShowAreaResult(X, bookmark.weights)
    net.ShowData(X, YData)
    