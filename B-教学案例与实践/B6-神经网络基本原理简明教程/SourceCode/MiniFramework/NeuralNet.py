# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.


import numpy as np
import time
import math
import os
import sys

from sklearn.metrics import r2_score

from MiniFramework.Layer import *
from MiniFramework.FullConnectionLayer import *
from MiniFramework.Parameters import *

class NeuralNet(object):
    def __init__(self, params, model_name):
        self.model_name = model_name
        self.params = params
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

    def __get_regular_cost_from_fc_layer(self, regularName):
        if regularName != RegularMethod.L1 and regularName != RegularMethod.L2:
            return 0

        regular_cost = 0
        for i in range(self.layer_count-1,-1,-1):
            layer = self.layer_list[i]
            if isinstance(layer, FcLayer):
                if regularName == RegularMethod.L1:
                    regular_cost += np.sum(np.abs(layer.weights.W))
                elif regularName == RegularMethod.L2:
                    regular_cost += np.sum(np.square(layer.weights.W))
            # end if
        # end for
        return regular_cost * self.params.lambd

    def __get_weights_from_fc_layer(self):
        weights = 0
        total = 0
        zeros = 0
        littles = 0
        for i in range(self.layer_count-1,-1,-1):
            layer = self.layer_list[i]
            if isinstance(layer, FcLayer):
                weights += np.sum(np.abs(layer.weights.W))
                zeros += len(np.where(np.abs(layer.weights.W)<=0.0001)[0])
                littles += len(np.where(np.abs(layer.weights.W)<=0.01)[0])
                total += np.size(layer.weights.W)
            # end if
        # end for
        return weights, zeros, littles, total

    # checkpoint=0.1 means will calculate the loss/accuracy every 10% in each epoch
    def train(self, dataReader, checkpoint=0.1, need_test=True):

        t0 = time.time()

        if self.params.regular == RegularMethod.EarlyStop:
           self.loss_history = CLossHistory(True, self.params.lambd)
        else:
           self.loss_history = CLossHistory()

        self.lossFunc = CLossFunction(self.params.loss_func)
        # if num_example=200, batch_size=10, then iteration=200/10=20
        if self.params.batch_size == -1 or self.params.batch_size > dataReader.num_train:
            self.params.batch_size = dataReader.num_train
        # end if
        max_iteration = math.ceil(dataReader.num_train / self.params.batch_size)
        checkpoint_iteration = (int)(max_iteration * checkpoint)
        need_stop = False
        for epoch in range(self.params.max_epoch):
            for iteration in range(max_iteration):
                # get x and y value for one sample
                batch_x, batch_y = dataReader.GetBatchTrainSamples(self.params.batch_size, iteration)
                # for optimizers which need pre-update weights
                if self.params.optimizer == OptimizerName.Nag:
                    self.__pre_update()
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

        weights, zeros, littles, total = self.__get_weights_from_fc_layer()
        print("total weights abs sum=", weights)
        print("total weights =", total)
        print("little weights =", littles)
        print("zero weights =", zeros)

        if need_test:
            print("testing...")
            accuracy = self.Test(dataReader, self.params.loss_func)
            print(accuracy)
        # end if

    def CheckErrorAndLoss(self, dataReader, train_x, train_y, epoch, total_iteration):
        print("epoch=%d, total_iteration=%d" %(epoch, total_iteration))

        # l1/l2 cost
        regular_cost = self.__get_regular_cost_from_fc_layer(self.params.regular)

        # calculate train loss
        self.__forward(train_x, train=False)
        loss_train = self.lossFunc.CheckLoss(train_y, self.output)
        loss_train = loss_train + regular_cost / train_x.shape[0]
        accuracy_train = self.__CalAccuracy(self.output, train_y, self.params.loss_func)
        print("loss_train=%.6f, accuracy_train=%f" %(loss_train, accuracy_train))

        # calculate validation loss
        vld_x, vld_y = dataReader.GetValidationSet()
        self.__forward(vld_x, train=False)
        loss_vld = self.lossFunc.CheckLoss(vld_y, self.output)
        loss_vld = loss_vld + regular_cost / vld_x.shape[0]
        accuracy_vld = self.__CalAccuracy(self.output, vld_y, self.params.loss_func)
        print("loss_valid=%.6f, accuracy_valid=%f" %(loss_vld, accuracy_vld))

        need_stop = self.loss_history.Add(epoch, total_iteration, loss_train, accuracy_train, loss_vld, accuracy_vld)
        if loss_vld <= self.params.eps:
            need_stop = True
        return need_stop

    def Test(self, dataReader, loss_func):
        x,y = dataReader.GetTestSet()
        self.__forward(x, train=False)
        correct = self.__CalAccuracy(self.output, y, loss_func)
        print(correct)
        return correct

    # mode: 1=fitting, 2=binary classifier, 3=multiple classifier
    def __CalAccuracy(self, a, y, loss_func):
        assert(a.shape == y.shape)
        m = a.shape[0]
        if loss_func == LossFunctionName.MSE:
            var = np.var(y)
            mse = np.sum((a-y)**2)/a.shape[0]
            r2 = 1 - mse / var
            return r2
        elif loss_func == LossFunctionName.CrossEntropy2:
            b = np.round(a)
            r = (b == y)
            correct = r.sum()
            return correct/m
        elif loss_func == LossFunctionName.CrossEntropy3:
            ra = np.argmax(a, axis=1)
            ry = np.argmax(y, axis=1)
            r = (ra == ry)
            correct = r.sum()
            return correct/m

    def inference(self, X):
        self.__forward(X, train=False)
        return self.output

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

    def ShowLossHistory(self, xmin=None, xmax=None, ymin=None, ymax=None):
        self.loss_history.ShowLossHistory(self.params, xmin, xmax, ymin, ymax)