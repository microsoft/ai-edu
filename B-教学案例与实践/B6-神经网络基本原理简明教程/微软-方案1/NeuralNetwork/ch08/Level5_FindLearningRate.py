# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
from LossFunction import * 
from Parameters import *
from Activations import *
from LearningRateSearcher import *
from DataOperator import *
from Level1_TwoLayer import *

x_data_name = "CurveX.dat"
y_data_name = "CurveY.dat"

def train(X, Y, params, loss_history, lr_searcher):
    num_example = X.shape[1]
    num_feature = X.shape[0]
    num_category = Y.shape[0]

    # W(num_category, num_feature), B(num_category, 1)
    W1, B1, W2, B2 = params.LoadSameInitialParameters()
    dict_weights = {"W1":W1, "B1":B1, "W2":W2, "B2":B2}

    # calculate loss to decide the stop condition
    loss = 0 
    lossFunc = CLossFunction(params.loss_func_name)

    lr, loop = lr_searcher.getFirstLearningRate()

    # if num_example=200, batch_size=10, then iteration=200/10=20
    max_iteration = (int)(params.num_example / params.batch_size)
    for epoch in range(params.max_epoch):
        for iteration in range(max_iteration):
            # get x and y value for one sample
            batch_x, batch_y = DataOperator.GetBatchSamples(X,Y,params.batch_size,iteration)
            # get z from x,y
            dict_cache = ForwardCalculationBatch(batch_x, dict_weights)
            # calculate gradient of w and b
            dict_grads = BackPropagationBatch(batch_x, batch_y, dict_cache, dict_weights)
            # update w,b
            dict_weights = UpdateWeights(dict_weights, dict_grads, lr)
        # end for            
        loss = lossFunc.CheckLoss(X, Y, dict_weights, ForwardCalculationBatch)
        print("epoch=%d, loss=%f, eta=%f" %(epoch,loss,lr))
        loss_history.AddLossHistory(loss, dict_weights, epoch, iteration)     

        if (epoch+1) % loop == 0:   # avoid when epoch==0
            lr, loop = lr_searcher.getNextLearningRate()
        # end if
        
        lr_searcher.addHistory(loss, lr)

        if lr >= 1:
            break
        if math.isnan(loss):
            break
        # end if
#        if loss < params.eps:
#            break
        # end if
    # end for
    
    return dict_weights

if __name__ == '__main__':

    X,Y = DataOperator.ReadData(x_data_name, y_data_name)
    num_example = X.shape[1]
    n_input, n_hidden, n_output = 1, 4, 1
    eta, batch_size, max_epoch = 0.0001, 10, 50000
    eps = 0.001
    init_method = InitialMethod.xavier
    lr_Searcher = CLearningRateSearcher()
#    looper = CLooper(0.0001,0.0001,100)
#    lr_Searcher.addLooper(looper)
#    looper = CLooper(0.001,0.001,100)
#    lr_Searcher.addLooper(looper)
#    looper = CLooper(0.01,0.01,100)
#    lr_Searcher.addLooper(looper)
    looper = CLooper(0.8,0.001,100)
    lr_Searcher.addLooper(looper)

    params = CParameters(num_example, n_input, n_output, n_hidden, eta, max_epoch, batch_size, LossFunctionName.MSE, eps, init_method)

    # SGD, MiniBatch, FullBatch
    loss_history = CLossHistory()
    dict_weights = train(X, Y, params, loss_history, lr_Searcher)

    lrs, losses = lr_Searcher.getLrLossHistory()
    plt.plot(np.log10(lrs), losses)
    plt.show()




