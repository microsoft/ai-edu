# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

"""
Version 2.0
"""

import numpy as np
import time
import math
import os
import sys

from HelperClass2.HyperParameters_2_0 import *
from HelperClass2.DataReader_2_0 import *
from HelperClass2.TrainingHistory_2_0 import *
from HelperClass2.LossFunction_1_1 import *
from HelperClass2.ActivatorFunction_2_0 import *
from HelperClass2.ClassifierFunction_2_0 import *
from HelperClass2.WeightsBias_1_0 import *

class NeuralNet_2_0(object):
    def __init__(self, hp, model_name):
        self.hp = hp
        self.model_name = model_name
        self.subfolder = os.getcwd() + "\\" + self.__create_subfolder()
        print(self.subfolder)

        self.wb1 = WeightsBias_1_0(self.hp.num_input, self.hp.num_hidden, self.hp.init_method, self.hp.eta)
        self.wb1.InitializeWeights(self.subfolder, False)
        self.wb2 = WeightsBias_1_0(self.hp.num_hidden, self.hp.num_output, self.hp.init_method, self.hp.eta)
        self.wb2.InitializeWeights(self.subfolder, False)

    def __create_subfolder(self):
        if self.model_name != None:
            path = self.model_name.strip()
            path = path.rstrip("\\")
            isExists = os.path.exists(path)
            if not isExists:
                os.makedirs(path)
            return path

    def forward(self, batch_x):
        # layer 1
        self.Z1 = np.dot(batch_x, self.wb1.W) + self.wb1.B
        self.A1 = Sigmoid().forward(self.Z1)
        # layer 2
        self.Z2 = np.dot(self.A1, self.wb2.W) + self.wb2.B
        if self.hp.net_type == NetType.BinaryClassifier:
            self.A2 = Logistic().forward(self.Z2)
        elif self.hp.net_type == NetType.MultipleClassifier:
            self.A2 = Softmax().forward(self.Z2)
        else:   # NetType.Fitting
            self.A2 = self.Z2
        #end if
        self.output = self.A2
        return self.output

    def backward(self, batch_x, batch_y, batch_a):
        # 批量下降，需要除以样本数量，否则会造成梯度爆炸
        m = batch_x.shape[0]
        # 第二层的梯度输入 公式5
        dZ2 = self.A2 - batch_y
        # 第二层的权重和偏移 公式6
        self.wb2.dW = np.dot(self.A1.T, dZ2)/m 
        # 公式7 对于多样本计算，需要在横轴上做sum，得到平均值
        self.wb2.dB = np.sum(dZ2, axis=0, keepdims=True)/m 
        # 第一层的梯度输入 公式8
        d1 = np.dot(dZ2, self.wb2.W.T) 
        # 第一层的dZ 公式10
        dZ1,_ = Sigmoid().backward(None, self.A1, d1)
        # 第一层的权重和偏移 公式11
        self.wb1.dW = np.dot(batch_x.T, dZ1)/m
        # 公式12 对于多样本计算，需要在横轴上做sum，得到平均值
        self.wb1.dB = np.sum(dZ1, axis=0, keepdims=True)/m 

    def update(self):
        self.wb1.Update()
        self.wb2.Update()

    def inference(self, x):
        self.forward(x)
        return self.output

    def train(self, dataReader, checkpoint, need_test):
        # calculate loss to decide the stop condition
        self.loss_trace = TrainingHistory_2_0()
        self.loss_func = LossFunction_1_1(self.hp.net_type)
        if self.hp.batch_size == -1:
            self.hp.batch_size = dataReader.num_train
        max_iteration = math.ceil(dataReader.num_train / self.hp.batch_size)
        checkpoint_iteration = (int)(max_iteration * checkpoint)
        need_stop = False
        for epoch in range(self.hp.max_epoch):
            #print("epoch=%d" %epoch)
            dataReader.Shuffle()
            for iteration in range(max_iteration):
                # get x and y value for one sample
                batch_x, batch_y = dataReader.GetBatchTrainSamples(self.hp.batch_size, iteration)
                # get z from x,y
                batch_a = self.forward(batch_x)
                # calculate gradient of w and b
                self.backward(batch_x, batch_y, batch_a)
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
        return correct

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
            correct = r.sum()
            return correct/m
        elif self.hp.net_type == NetType.MultipleClassifier:
            ra = np.argmax(a, axis=1)
            ry = np.argmax(y, axis=1)
            r = (ra == ry)
            correct = r.sum()
            return correct/m

    def SaveResult(self):
        self.wb1.SaveResultValue(self.subfolder, "wb1")
        self.wb2.SaveResultValue(self.subfolder, "wb2")

    def LoadResult(self):
        self.wb1.LoadResultValue(self.subfolder, "wb1")
        self.wb2.LoadResultValue(self.subfolder, "wb2")

    def ShowTrainingHistory(self):
        self.loss_trace.ShowLossHistory(self.hp)

    def GetTrainingHistory(self):
        return self.loss_trace
