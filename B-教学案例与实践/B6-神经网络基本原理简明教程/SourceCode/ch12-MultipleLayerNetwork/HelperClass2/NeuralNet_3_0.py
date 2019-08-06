# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

"""
Version 3.0
what's new:
- support 3 layers
"""

import numpy as np
import time
import math
import os
import sys

from HelperClass2.HyperParameters_3_0 import *
from HelperClass2.MnistImageDataReader import *
from HelperClass2.TrainingHistory_2_3 import *
from HelperClass2.LossFunction_1_1 import *
from HelperClass2.ActivatorFunction_2_0 import *
from HelperClass2.ClassifierFunction_2_0 import *
from HelperClass2.WeightsBias_1_0 import *

class NeuralNet_3_0(object):
    def __init__(self, hp, model_name):
        self.hp = hp
        self.model_name = model_name
        self.subfolder = os.getcwd() + "/" + self.__create_subfolder()
        print(self.subfolder)

        self.wb1 = WeightsBias_1_0(self.hp.num_input, self.hp.num_hidden1, self.hp.init_method, self.hp.eta)
        self.wb1.InitializeWeights(self.subfolder, False)
        self.wb2 = WeightsBias_1_0(self.hp.num_hidden1, self.hp.num_hidden2, self.hp.init_method, self.hp.eta)
        self.wb2.InitializeWeights(self.subfolder, False)
        self.wb3 = WeightsBias_1_0(self.hp.num_hidden2, self.hp.num_output, self.hp.init_method, self.hp.eta)
        self.wb3.InitializeWeights(self.subfolder, False)

    def __create_subfolder(self):
        if self.model_name != None:
            path = self.model_name.strip()
            path = path.rstrip("/")
            isExists = os.path.exists(path)
            if not isExists:
                os.makedirs(path)
            return path

    def forward(self, batch_x):
        # 公式1
        self.Z1 = np.dot(batch_x, self.wb1.W) + self.wb1.B
        # 公式2
        self.A1 = Sigmoid().forward(self.Z1)
        #self.A1 = 1.0 / (1.0 + np.exp(-self.Z1))
        # 公式3
        self.Z2 = np.dot(self.A1, self.wb2.W) + self.wb2.B
        # 公式4
        self.A2 = Tanh().forward(self.Z2)
        #self.A2 = 2.0 / (1.0 + np.exp(-2*self.Z2)) - 1.0
        # 公式5
        self.Z3 = np.dot(self.A2, self.wb3.W) + self.wb3.B
        # 公式6
        if self.hp.net_type == NetType.BinaryClassifier:
            self.A3 = Logistic().forward(self.Z3)
        elif self.hp.net_type == NetType.MultipleClassifier:
            self.A3 = Softmax().forward(self.Z3)
        else:   # NetType.Fitting
            self.A3 = self.Z3
        #end if
        self.output = self.A3

    def backward(self, batch_x, batch_y):
        # 批量下降，需要除以样本数量，否则会造成梯度爆炸
        m = batch_x.shape[0]

        # 第三层的梯度输入 公式7
        dZ3 = self.output - batch_y
        # 公式8
        self.wb3.dW = np.dot(self.A2.T, dZ3)/m
        # 公式9
        self.wb3.dB = np.sum(dZ3, axis=0, keepdims=True)/m

        # 第二层的梯度输入 公式10
        dA2 = np.dot(dZ3, self.wb3.W.T)
        # 公式11
        dZ2,_ = Tanh().backward(None, self.A2, dA2)
        #da = 1 - np.multiply(self.A2, self.A2)
        #dZ2 = np.multiply(dA2, da)
        # 公式12
        self.wb2.dW = np.dot(self.A1.T, dZ2)/m
        # 公式13
        self.wb2.dB = np.sum(dZ2, axis=0, keepdims=True)/m

        # 第一层的梯度输入 公式8
        dA1 = np.dot(dZ2, self.wb2.W.T) 
        # 第一层的dZ 公式10
        dZ1,_ = Sigmoid().backward(None, self.A1, dA1)
        #da = np.multiply(self.A1, 1-self.A1)
        #dZ1 = np.multiply(dA1, da)

        # 第一层的权重和偏移 公式11
        self.wb1.dW = np.dot(batch_x.T, dZ1)/m
        self.wb1.dB = np.sum(dZ1, axis=0, keepdims=True)/m

    def update(self):
        self.wb1.Update()
        self.wb2.Update()
        self.wb3.Update()

    def inference(self, x):
        self.forward(x)
        return self.output

    def train(self, dataReader, checkpoint, need_test):
        t0 = time.time()
        # calculate loss to decide the stop condition
        self.loss_trace = TrainingHistory_2_3()
        self.loss_func = LossFunction_1_1(self.hp.net_type)
        loss = 10
        if self.hp.batch_size == -1:
            self.hp.batch_size = dataReader.num_train
        max_iteration = math.ceil(dataReader.num_train / self.hp.batch_size)
        checkpoint_iteration = (int)(max_iteration * checkpoint)
        need_stop = False
        for epoch in range(self.hp.max_epoch):
            dataReader.Shuffle()
            for iteration in range(max_iteration):
                # get x and y value for one sample
                batch_x, batch_y = dataReader.GetBatchTrainSamples(self.hp.batch_size, iteration)
                # get z from x,y
                self.forward(batch_x)
                # calculate gradient of w and b
                self.backward(batch_x, batch_y)
                # update w,b
                self.update()

                total_iteration = epoch * max_iteration + iteration
                if (total_iteration+1) % checkpoint_iteration == 0:
                    need_stop = self.CheckErrorAndLoss(dataReader, batch_x, batch_y, epoch, total_iteration)
                    if need_stop:
                        break                
                    #end if
                #end if
            # end for
            if need_stop:
                break
        # end for
        self.SaveResult()
        
        t1 = time.time()
        print("time used:", t1 - t0)

        #self.CheckErrorAndLoss(dataReader, batch_x, batch_y, epoch, total_iteration)
        if need_test:
            print("testing...")
            accuracy = self.Test(dataReader)
            print(accuracy)
        # end if

    def CheckErrorAndLoss(self, dataReader, train_x, train_y, epoch, total_iteration):
        print("epoch=%d, total_iteration=%d" %(epoch, total_iteration))

        # calculate train loss
        self.forward(train_x)
        loss_train = self.loss_func.CheckLoss(self.output, train_y)
        accuracy_train = self.__CalAccuracy(self.output, train_y)
        print("loss_train=%.6f, accuracy_train=%f" %(loss_train, accuracy_train))

        # calculate validation loss
        vld_x, vld_y = dataReader.GetValidationSet()
        self.forward(vld_x)
        loss_vld = self.loss_func.CheckLoss(self.output, vld_y)
        accuracy_vld = self.__CalAccuracy(self.output, vld_y)
        print("loss_valid=%.6f, accuracy_valid=%f" %(loss_vld, accuracy_vld))

        need_stop = self.loss_trace.Add(epoch, total_iteration, loss_train, accuracy_train, loss_vld, accuracy_vld)
        if loss_vld <= self.hp.eps:
            need_stop = True
        return need_stop

    def Test(self, dataReader):
        x,y = dataReader.GetTestSet()
        self.forward(x)
        correct = self.__CalAccuracy(self.output, y)
        print(correct)

    def __CalAccuracy(self, a, y):
        assert(a.shape == y.shape)
        m = a.shape[0]
        if self.hp.net_type == NetType.Fitting:
            var = np.var(y)
            mse = np.sum((a-y)**2)/m
            r2 = 1 - mse / var
            return r2
        elif self.hp.net_type == NetType.BinaryClassifier:
            b = np.round(a)
            r = (b == y)
            correct = np.sum(r)
            return correct/m
        elif self.hp.net_type == NetType.MultipleClassifier:
            ra = np.argmax(a, axis=1)
            ry = np.argmax(y, axis=1)
            r = (ra == ry)
            correct = np.sum(r)
            return correct/m

    def SaveResult(self):
        self.wb1.SaveResultValue(self.subfolder, "wb1")
        self.wb2.SaveResultValue(self.subfolder, "wb2")
        self.wb3.SaveResultValue(self.subfolder, "wb3")

    def LoadResult(self):
        self.wb1.LoadResultValue(self.subfolder, "wb1")
        self.wb2.LoadResultValue(self.subfolder, "wb2")
        self.wb3.LoadResultValue(self.subfolder, "wb3")

    def ShowTrainingHistory(self, xcoord):
        self.loss_trace.ShowLossHistory(self.hp, xcoord)

    def GetTrainingTrace(self):
        return self.loss_trace

    def GetEpochNumber(self):
        return self.loss_trace.GetEpochNumber()

    def GetLatestAverageLoss(self, count=10):
        return self.loss_trace.GetLatestAverageLoss(count)

    def DumpLossHistory(self, filename):
        return self.loss_trace.Dump(filename)
