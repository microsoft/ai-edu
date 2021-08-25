# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import time
import os
import math

from MiniFramework.EnumDef_6_0 import *
from MiniFramework.Layer import *
from MiniFramework.FCLayer_2_0 import *
from MiniFramework.DropoutLayer import *
from MiniFramework.BatchNormLayer import *
from MiniFramework.ActivationLayer import *
from MiniFramework.ClassificationLayer import *
from MiniFramework.ConvLayer import *
from MiniFramework.PoolingLayer import *
from MiniFramework.HyperParameters_4_2 import *
from MiniFramework.TrainingHistory_3_0 import *
from MiniFramework.LossFunction_1_1 import *

from MiniFramework.DataReader_2_0 import *

class NeuralNet_4_2(object):
    def __init__(self, params, model_name):
        self.model_name = model_name
        self.hp = params
        self.layer_list = []
        self.output = None
        self.layer_count = 0
        self.subfolder = os.getcwd() + "/" + self.__create_subfolder()
        print(self.subfolder)
        self.accuracy = 0

    def __create_subfolder(self):
        if self.model_name != None:
            path = self.model_name.strip()
            path = path.rstrip("/")
            isExists = os.path.exists(path)
            if not isExists:
                os.makedirs(path)
            return path

    def add_layer(self, layer, name):
        layer.initialize(self.subfolder, name)
        self.layer_list.append(layer)
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

    def __get_regular_cost_from_fc_layer(self, regularName):
        if regularName != RegularMethod.L1 and regularName != RegularMethod.L2:
            return 0

        regular_cost = 0
        for i in range(self.layer_count-1,-1,-1):
            layer = self.layer_list[i]
            if isinstance(layer, FcLayer_2_0):
                if regularName == RegularMethod.L1:
                    regular_cost += np.sum(np.abs(layer.wb.W))
                elif regularName == RegularMethod.L2:
                    regular_cost += np.sum(np.square(layer.wb.W))
            # end if
        # end for
        return regular_cost * self.hp.regular_value

    def __check_weights_from_fc_layer(self):
        weights = 0
        total = 0
        zeros = 0
        littles = 0
        for i in range(self.layer_count-1,-1,-1):
            layer = self.layer_list[i]
            if isinstance(layer, FcLayer_2_0):
                weights += np.sum(np.abs(layer.wb.W))
                zeros += len(np.where(np.abs(layer.wb.W)<=0.0001)[0])
                littles += len(np.where(np.abs(layer.wb.W)<=0.01)[0])
                total += np.size(layer.wb.W)
            # end if
        # end for
        print("total weights abs sum=", weights)
        print("total weights =", total)
        print("little weights =", littles)
        print("zero weights =", zeros)

    # checkpoint=0.1 means will calculate the loss/accuracy every 10% in each epoch
    def train(self, dataReader, checkpoint=0.1, need_test=True):
        t0 = time.time()
        self.lossFunc = LossFunction_1_1(self.hp.net_type)
        if self.hp.regular_name == RegularMethod.EarlyStop:
            self.loss_trace = TrainingHistory_3_0(True, self.hp.regular_value)
        else:
           self.loss_trace = TrainingHistory_3_0()

        if self.hp.batch_size == -1 or self.hp.batch_size > dataReader.num_train:
            self.hp.batch_size = dataReader.num_train
        # end if
        max_iteration = math.ceil(dataReader.num_train / self.hp.batch_size)
        checkpoint_iteration = (int)(math.ceil(max_iteration * checkpoint))
        need_stop = False
        for epoch in range(self.hp.max_epoch):
            dataReader.Shuffle()
            for iteration in range(max_iteration):
                # get x and y value for one sample
                batch_x, batch_y = dataReader.GetBatchTrainSamples(self.hp.batch_size, iteration)
                # for optimizers which need pre-update weights
                if self.hp.optimizer_name == OptimizerName.Nag:
                    self.__pre_update()
                # get z from x,y
                self.__forward(batch_x, train=True)
                # calculate gradient of w and b
                self.__backward(batch_x, batch_y)
                # final update w,b
                self.__update()
                
                total_iteration = epoch * max_iteration + iteration               
                if (total_iteration+1) % checkpoint_iteration == 0:
                    need_stop = self.CheckErrorAndLoss(dataReader, batch_x, batch_y, epoch, total_iteration)
                    if need_stop:
                        break                
                #end if
            # end for
            if need_stop:
                break
            # end if
        # end for

        t1 = time.time()
        print("time used:", t1 - t0)

        if need_test:
            print("testing...")
            self.accuracy = self.Test(dataReader)
        # end if

    def CheckErrorAndLoss(self, dataReader, train_x, train_y, epoch, total_iteration):
        print("epoch=%d, total_iteration=%d" %(epoch, total_iteration))

        # l1/l2 cost
        regular_cost = self.__get_regular_cost_from_fc_layer(self.hp.regular_name)

        # calculate train loss
        self.__forward(train_x, train=False)
        loss_train, accuracy_train = self.lossFunc.CheckLoss(self.output, train_y)
        loss_train = loss_train + regular_cost / train_x.shape[0]
        print("loss_train=%.6f, accuracy_train=%f" %(loss_train, accuracy_train))

        # calculate validation loss
        vld_x, vld_y = dataReader.GetValidationSet()
        self.__forward(vld_x, train=False)
        loss_vld, accuracy_vld = self.lossFunc.CheckLoss(self.output, vld_y)
        loss_vld = loss_vld + regular_cost / vld_x.shape[0]
        print("loss_valid=%.6f, accuracy_valid=%f" %(loss_vld, accuracy_vld))

        if self.loss_trace.IsMaximum(accuracy_vld):
            self.save_parameters()

        # end if
        need_stop = self.loss_trace.Add(epoch, total_iteration, loss_train, accuracy_train, loss_vld, accuracy_vld, self.hp.stopper)
        if self.hp.stopper is not None:
            if self.hp.stopper.stop_condition == StopCondition.StopLoss and loss_vld <= self.hp.stopper.stop_value:
                need_stop = True
        return need_stop
        
    def Test(self, dataReader):
        x,y = dataReader.GetTestSet()
        self.__forward(x, train=False)
        _, correct1 = self.lossFunc.CheckLoss(self.output, y)
        print(correct1)

        self.load_parameters()
        self.__forward(x, train=False)
        _, correct2 = self.lossFunc.CheckLoss(self.output, y)
        print(correct2)

        return correct2

    # save weights value when got low loss than before
    def save_parameters(self):
        print("save parameters")
        for i in range(self.layer_count):
            layer = self.layer_list[i]
            layer.save_parameters()

    # load weights for the most low loss moment
    def load_parameters(self):
        print("load parameters")
        for i in range(self.layer_count):
            layer = self.layer_list[i]
            layer.load_parameters()

    def ShowLossHistory(self, xcoor, xmin=None, xmax=None, ymin=None, ymax=None):
        title = str.format("{0},accuracy={1:.4f}", self.hp.toString(), self.accuracy)
        self.loss_trace.ShowLossHistory(title, xcoor, xmin, xmax, ymin, ymax)
