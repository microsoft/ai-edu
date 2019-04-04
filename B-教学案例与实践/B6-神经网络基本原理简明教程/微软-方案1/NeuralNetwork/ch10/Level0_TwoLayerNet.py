# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math

from LossFunction import * 
from DataReader import * 
from WeightsBias import *
from Parameters import *
from Activators import *

class NetType(Enum):
    Fitting = 1,
    BinaryClassifier = 2,
    MultipleClassifier = 3

class TwoLayerNet(object):
    def __init__(self, net_type):
        self.net_type = net_type
        if self.net_type == NetType.Fitting:
            self.loss_func_name = LossFunctionName.MSE
            self.forward = self.ForwardCalculationBatch1
        elif self.net_type == NetType.BinaryClassifier:
            self.loss_func_name = LossFunctionName.CrossEntropy2
            self.forward = self.ForwardCalculationBatch2
        elif self.net_type == NetType.MultipleClassifier:
            self.loss_func_name = LossFunctionName.CrossEntropy3
            self.forward = self.ForwardCalculationBatch3


    # 用于拟合
    def ForwardCalculationBatch1(self, batch_x, wb1, wb2):
        # layer 1
        Z1 = np.dot(wb1.W, batch_x) + wb1.B
        A1 = Sigmoid().forward(Z1)
        # layer 2
        Z2 = np.dot(wb2.W, A1) + wb2.B
        A2 = Z2
        # keep cache for backward
        dict_cache ={"Z2": Z2, "A1": A1, "A2": A2, "Output": A2}
        return dict_cache

    # 用于二分类
    def ForwardCalculationBatch2(self, batch_x, wb1, wb2):
        # layer 1
        Z1 = np.dot(wb1.W, batch_x) + wb1.B
        A1 = Sigmoid().forward(Z1)
        # layer 2
        Z2 = np.dot(wb2.W, A1) + wb2.B
        A2 = Sigmoid().forward(Z2)
        # keep cache for backward
        dict_cache ={"Z2": Z2, "A1": A1, "A2": A2, "Output": A2}
        return dict_cache

    # 用于多分类
    def ForwardCalculationBatch3(self, batch_x, wb1, wb2):
        # layer 1
        Z1 = np.dot(wb1.W, batch_x) + wb1.B
        A1 = Sigmoid().forward(Z1)
        # layer 2
        Z2 = np.dot(wb2.W, A1) + wb2.B
        A2 = Softmax().forward(Z2)
        # keep cache for backward
        dict_cache ={"Z2": Z2, "A1": A1, "A2": A2, "Output": A2}
        return dict_cache

    def BackPropagationBatch(self, batch_x, batch_y, dict_cache, wb1, wb2):
        # 批量下降，需要除以样本数量，否则会造成梯度爆炸
        m = batch_x.shape[1]
        # 取出缓存值
        A1 = dict_cache["A1"]
        A2 = dict_cache["A2"]
        # 第二层的梯度输入
        dZ2 = A2 - batch_y 
        # 第二层的权重和偏移
        wb2.dW = np.dot(dZ2, A1.T)/m  
        wb2.dB = np.sum(dZ2, axis=1, keepdims=True)/m 
        # 第一层的梯度输入
        d1 = np.dot(wb2.W.T, dZ2) 
        # 第一层的dZ
        #dZ1 = d1 * A1 * (1-A1)
        dZ1,_ = Sigmoid().backward(None, A1, d1)  
        # 第一层的权重和偏移
        wb1.dW = np.dot(dZ1, batch_x.T)/m  
        wb1.dB = np.sum(dZ1, axis=1, keepdims=True)/m  

    def UpdateWeights(self, wb1, wb2):
        wb1.Update()
        wb2.Update()

    def train(self, dataReader, params, loss_history):
        # initialize weights and bias
        wb1 = WeightsBias(params.num_input, params.num_hidden, params.eta, params.init_method, params.optimizer_name)
        wb1.InitializeWeights(False)
        wb2 = WeightsBias(params.num_hidden, params.num_output, params.eta, params.init_method, params.optimizer_name)
        wb2.InitializeWeights(False)
        # calculate loss to decide the stop condition
        loss = 0 

        lossFunc = CLossFunction(self.loss_func_name)

        # if num_example=200, batch_size=10, then iteration=200/10=20
        if params.batch_size == -1 or params.batch_size > dataReader.num_example: # full batch
            params.batch_size = dataReader.num_example
        max_iteration = (int)(dataReader.num_example / params.batch_size)
        for epoch in range(params.max_epoch):
            for iteration in range(max_iteration):

                if params.optimizer_name == OptimizerName.Nag:
                    wb1.pre_Update()
                    wb2.pre_Update()

                # get x and y value for one sample
                batch_x, batch_y = dataReader.GetBatchSamples(params.batch_size,iteration)
                # get z from x,y
                dict_cache = self.forward(batch_x, wb1, wb2)
                # calculate gradient of w and b
                self.BackPropagationBatch(batch_x, batch_y, dict_cache, wb1, wb2)
                # update w,b
                self.UpdateWeights(wb1, wb2)
            # end for            
            # calculate loss for this batch
            output = self.forward(dataReader.X, wb1, wb2)
            loss = lossFunc.CheckLoss(dataReader.Y, output["Output"])
            print("epoch=%d, loss=%f" %(epoch,loss))
            loss_history.AddLossHistory(loss, epoch, iteration, wb1, wb2)            
            if math.isnan(loss):
                break
            # end if
            if loss < params.eps:
                break
            # end if
        # end for
        return wb1, wb2
    # end def

  
# end class

