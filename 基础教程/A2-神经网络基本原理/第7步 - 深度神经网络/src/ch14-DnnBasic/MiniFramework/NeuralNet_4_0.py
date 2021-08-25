# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

"""
Version 4.0
what's new?
- totally different than previsous version, support building-block
"""

import numpy as np
import time
import math
import os
import sys

from MiniFramework.DataReader_2_0 import *
from MiniFramework.Layer import *
from MiniFramework.FullConnectionLayer_1_0 import *
from MiniFramework.HyperParameters_4_0 import *
from MiniFramework.LossFunction_1_1 import *
from MiniFramework.TrainingHistory_2_4 import *

class NeuralNet_4_0(object):
    def __init__(self, params, model_name):
        self.model_name = model_name
        self.hp = params
        self.layer_list = []
        self.layer_name = []
        self.output = None
        self.layer_count = 0
        self.subfolder = os.getcwd() + "\\" + self.__create_subfolder()
        print(self.subfolder)

    def __create_subfolder(self):
        if self.model_name != None:
            path = self.model_name.strip()
            path = path.rstrip("\\")
            isExists = os.path.exists(path)
            if not isExists:
                os.makedirs(path)
            return path

    def add_layer(self, layer, name=""):
        layer.initialize(self.subfolder)
        self.layer_list.append(layer)
        self.layer_name.append(name)
        self.layer_count += 1

    def __forward(self, X, train=True):
        input = X
        for i in range(self.layer_count):
            layer = self.layer_list[i]
            output = layer.forward(input, train)
            input = output
        # end for
        self.output = output
        return self.output

    def inference(self, X):
        output = self.__forward(X, train=False)
        return output

    def __backward(self, X, Y):
        delta_in = self.output - Y
        for i in range(self.layer_count-1,-1,-1):
            layer = self.layer_list[i]
            delta_out = layer.backward(delta_in, i)
            # move back to previous layer
            delta_in = delta_out

    def __pre_update(self):
        for i in range(self.layer_count-1,-1,-1):
            layer = self.layer_list[i]
            layer.pre_update()

    def __update(self):
        for i in range(self.layer_count-1,-1,-1):
            layer = self.layer_list[i]
            layer.update()

    # checkpoint=0.1 means will calculate the loss/accuracy every 10% in each epoch
    def train(self, dataReader, checkpoint=0.1, need_test=True):
        t0 = time.time()
        self.loss_trace = TrainingHistory_2_4()
        self.lossFunc = LossFunction_1_1(self.hp.net_type)
        # if num_example=200, batch_size=10, then iteration=200/10=20
        if self.hp.batch_size == -1 or self.hp.batch_size > dataReader.num_train:
            self.hp.batch_size = dataReader.num_train
        # end if
        max_iteration = math.ceil(dataReader.num_train / self.hp.batch_size)
        checkpoint_iteration = (int)(max_iteration * checkpoint)
        need_stop = False
        for epoch in range(self.hp.max_epoch):
            for iteration in range(max_iteration):
                # get x and y value for one sample
                batch_x, batch_y = dataReader.GetBatchTrainSamples(self.hp.batch_size, iteration)
                # get z from x,y
                self.__forward(batch_x, train=True)
                # calculate gradient of w and b
                self.__backward(batch_x, batch_y)
                # final update w,b
                self.__update()
                
                total_iteration = epoch * max_iteration + iteration               
                if (total_iteration+1) % checkpoint_iteration == 0:
                    #self.save_parameters()
                    need_stop = self.CheckErrorAndLoss(dataReader, batch_x, batch_y, epoch, total_iteration)
                    if need_stop:
                        break                
                #end if
            # end for
            #self.save_parameters()  # 这里会显著降低性能，因为频繁的磁盘操作，而且可能会有文件读写失败
            dataReader.Shuffle()
            if need_stop:
                break
            # end if
        # end for
        self.CheckErrorAndLoss(dataReader, batch_x, batch_y, epoch, total_iteration)

        t1 = time.time()
        print("time used:", t1 - t0)

        self.save_parameters()

        if need_test:
            print("testing...")
            accuracy = self.Test(dataReader)
            print(accuracy)
        # end if

    def CheckErrorAndLoss(self, dataReader, train_x, train_y, epoch, total_iteration):
        print("epoch=%d, total_iteration=%d" %(epoch, total_iteration))

        # calculate train loss
        self.__forward(train_x, train=False)
        loss_train = self.lossFunc.CheckLoss(self.output, train_y)
        loss_train = loss_train# + regular_cost / train_x.shape[0]
        accuracy_train = self.__CalAccuracy(self.output, train_y)
        print("loss_train=%.6f, accuracy_train=%f" %(loss_train, accuracy_train))

        # calculate validation loss
        vld_x, vld_y = dataReader.GetValidationSet()
        self.__forward(vld_x, train=False)
        loss_vld = self.lossFunc.CheckLoss(self.output, vld_y)
        loss_vld = loss_vld #+ regular_cost / vld_x.shape[0]
        accuracy_vld = self.__CalAccuracy(self.output, vld_y)
        print("loss_valid=%.6f, accuracy_valid=%f" %(loss_vld, accuracy_vld))

        need_stop = self.loss_trace.Add(epoch, total_iteration, loss_train, accuracy_train, loss_vld, accuracy_vld, self.hp.stopper)
        if self.hp.stopper.stop_condition == StopCondition.StopLoss and loss_vld <= self.hp.stopper.stop_value:
            need_stop = True
        return need_stop

    def Test(self, dataReader):
        x,y = dataReader.GetTestSet()
        self.__forward(x, train=False)
        correct = self.__CalAccuracy(self.output, y)
        return correct

    def __CalAccuracy(self, a, y):
        assert(a.shape == y.shape)
        m = a.shape[0]
        if self.hp.net_type == NetType.Fitting:
            var = np.var(y)
            mse = np.sum((a-y)**2)/a.shape[0]
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

    # save weights value when got low loss than before
    def save_parameters(self):
        print("save parameters")
        for i in range(self.layer_count):
            layer = self.layer_list[i]
            name = self.layer_name[i]
            layer.save_parameters(self.subfolder, name)

    # load weights for the most low loss moment
    def load_parameters(self):
        print("load parameters")
        for i in range(self.layer_count):
            layer = self.layer_list[i]
            name = self.layer_name[i]
            layer.load_parameters(self.subfolder, name)

    def ShowLossHistory(self, xcoord=XCoordinate.Epoch, xmin=None, xmax=None, ymin=None, ymax=None):
        self.loss_trace.ShowLossHistory(self.hp.toString(), xcoord, xmin, xmax, ymin, ymax)
            
    def GetTrainingTrace(self):
        return self.loss_trace

    def GetEpochNumber(self):
        return self.loss_trace.GetEpochNumber()

    def GetLatestAverageLoss(self, count=10):
        return self.loss_trace.GetLatestAverageLoss(count)
