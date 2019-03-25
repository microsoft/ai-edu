# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
from LossFunction import * 
from Activations import *
from DataReader import * 
from GDOptimizer import *
from WeightsBias import *

class CTwoLayerNet(object):

    def ForwardCalculationBatch(self, batch_x, wbs):
        # layer 1
        Z1 = np.dot(wbs.W1, batch_x) + wbs.B1
        A1 = CSigmoid().forward(Z1)
        # layer 2
        Z2 = np.dot(wbs.W2, A1) + wbs.B2
        A2 = CSoftmax().forward(Z2)

        # keep cache for backward
        dict_cache ={"Z2": Z2, "A1": A1, "A2": A2, "Output": A2}
        return dict_cache

    def BackPropagationBatch(self, batch_x, batch_y, dict_cache, wbs):
        # 批量下降，需要除以样本数量，否则会造成梯度爆炸
        m = batch_x.shape[1]
        # 取出缓存值
        A1 = dict_cache["A1"]
        A2 = dict_cache["A2"]
        # 第二层的梯度输入
        dZ2 = A2 - batch_y
        # 第二层的权重和偏移
        wbs.dW2 = np.dot(dZ2, A1.T)/m
        wbs.dB2 = np.sum(dZ2, axis=1, keepdims=True)/m
        # 第一层的梯度输入
        dA1 = np.dot(wbs.W2.T, dZ2)
        dA1_Z1 = np.multiply(A1, 1 - A1)    # for Sigmoid derivative
        dZ1 = np.multiply(dA1, dA1_Z1)
        # 第一层的权重和偏移
        wbs.dW1 = np.dot(dZ1, batch_x.T)/m
        wbs.dB1 = np.sum(dZ1, axis=1, keepdims=True)/m

    def train(self, dataReader, params, loss_history):
        optimizer = GDOptimizerFactory.CreateOptimizer(params.optimizer_name)
        wbs = WeightsBias(params)
        wbs.InitializeWeights(False)

        # calculate loss to decide the stop condition
        loss = 0 
        lossFunc = CLossFunction(params.loss_func_name)

        # if num_example=200, batch_size=10, then iteration=200/10=20
        max_iteration = (int)(dataReader.num_example / params.batch_size)
        for epoch in range(params.max_epoch):
            for iteration in range(max_iteration):
                # get x and y value for one sample
                batch_x, batch_y = dataReader.GetBatchSamples(params.batch_size, iteration)

                if params.optimizer_name == OptimizerName.O_Nag:
                    wbs.pre_Update()

                # get z from x,y
                dict_cache = self.ForwardCalculationBatch(batch_x, wbs)
                # calculate gradient of w and b
                self.BackPropagationBatch(batch_x, batch_y, dict_cache, wbs)
                # update w,b
                wbs.Update()
            # end for            
            # calculate loss for this batch
            loss = lossFunc.CheckLoss(dataReader.X, dataReader.Y, wbs, self.ForwardCalculationBatch)
            print("epoch=%d, loss=%f" %(epoch,loss))
            loss_history.AddLossHistory(loss, wbs.GetWeightsBiasAsDict(), epoch, iteration)
            if math.isnan(loss):
                break
            # end if
            if loss < params.eps:
                break
            # end if
        # end for
        return wbs
    # end def

    def ShowAreaResult(self, X, dict_weights):
        count = 50
        x1 = np.linspace(0,1,count)
        x2 = np.linspace(0,1,count)
        for i in range(count):
            for j in range(count):
                x = np.array([x1[i],x2[j]]).reshape(2,1)
                dict_cache = self.ForwardCalculationBatch(x, dict_weights)
                output = dict_cache["Output"]
                r = np.argmax(output, axis=0)
                if r == 0:
                    plt.plot(x[0,0], x[1,0], 's', c='m')
                elif r == 1:
                    plt.plot(x[0,0], x[1,0], 's', c='y')
                # end if
            # end for
        # end for
    #end def

    def ShowData(self, X, Y):
        for i in range(X.shape[1]):
            if Y[0,i] == 1:
                plt.plot(X[0,i], X[1,i], '^', c='g')
            elif Y[0,i] == 2:
                plt.plot(X[0,i], X[1,i], 'x', c='r')
            elif Y[0,i] == 3:
                plt.plot(X[0,i], X[1,i], '.', c='b')
            # end if
        # end for
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()

# end class

# this class is for two-layer NN only
class CParameters(object):
    def __init__(self, n_input=1, n_output=1, n_hidden=4, 
                 eta=0.1, max_epoch=10000, batch_size=5, eps=0.001,
                 lossFuncName=LossFunctionName.MSE, 
                 initMethod=InitialMethod.Zero, 
                 optimizerName=OptimizerName.O_SGD):

        self.num_input = n_input
        self.num_output = n_output
        self.num_hidden = n_hidden
        self.eta = eta
        self.max_epoch = max_epoch
        # if batch_size == -1, it is FullBatch
        if batch_size == -1:
            self.batch_size = self.num_example
        else:
            self.batch_size = batch_size
        # end if
        self.loss_func_name = lossFuncName
        self.eps = eps
        self.init_method = initMethod
        self.optimizer_name = optimizerName
