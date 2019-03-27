# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
from LossFunction import * 
from DataReader import * 
from GDOptimizer import *
from WeightsBias import *
from Parameters import *

class TwoLayerNet(object):

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
                # for optimizers which need pre-update weights
                if params.optimizer_name == OptimizerName.Nag:
                    wbs.pre_Update()

                # get z from x,y
                dict_cache = self.ForwardCalculationBatch(batch_x, wbs)
                # calculate gradient of w and b
                self.BackPropagationBatch(batch_x, batch_y, dict_cache, wbs)
                # final update w,b
                wbs.Update()
            # end for            
            # calculate loss for this batch
            dict_cache = self.ForwardCalculationBatch(dataReader.X, wbs)
            loss = lossFunc.CheckLoss(dataReader.Y, dict_cache["Output"])
            print("epoch=%d, loss=%f" %(epoch,loss))
            loss_history.AddLossHistory(loss, wbs.GetWeightsBiasAsDict(), epoch, iteration)
            if math.isnan(loss):
                break
            # end if
            if loss < params.eps:
                break
            # end if
            dataReader.Shuffle()
        # end for
        return wbs
    # end def
# end class

