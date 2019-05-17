# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.


import numpy as np
import time

from MiniFramework.Layer import *
from MiniFramework.FullConnectionLayer import *
from MiniFramework.Parameters import *

class NeuralNet(object):
    def __init__(self, params):
        self.params = params
        self.layer_list = []
        self.layer_name = []
        self.output = None
        self.layer_count = 0

    def add_layer(self, layer, name=""):
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

    def __get_layer_index(self, idx):
        if self.layer_count == 1:
            return LayerIndexFlags.SingleLayer
        else:
            if idx == self.layer_count - 1:
                return LayerIndexFlags.LastLayer
            elif idx == 0:
                return LayerIndexFlags.FirstLayer
            else:
                return LayerIndexFlags.MiddleLayer

    # checkpoint=0.1 means will calculate the loss/accuracy every 10% in each epoch
    def train(self, dataReader, checkpoint=0.1, need_test=True):

        t0 = time.time()

        self.loss_history = CLossHistory()
        loss = 0 
        self.lossFunc = CLossFunction(self.params.loss_func_name)
        # if num_example=200, batch_size=10, then iteration=200/10=20
        if self.params.batch_size == -1 or self.params.batch_size > dataReader.num_train:
            self.params.batch_size = dataReader.num_train
        # end if
        max_iteration = dataReader.num_train // self.params.batch_size
        checkpoint_iteration = (int)(max_iteration * checkpoint)
        for epoch in range(self.params.max_epoch):
            for iteration in range(max_iteration):
                # get x and y value for one sample
                batch_x, batch_y = dataReader.GetBatchTrainSamples(self.params.batch_size, iteration)
                # for optimizers which need pre-update weights
                if self.params.optimizer_name == OptimizerName.Nag:
                    self.__pre_update()
                # get z from x,y
                self.__forward(batch_x, train=True)
                # calculate gradient of w and b
                self.__backward(batch_x, batch_y)
                # final update w,b
                self.__update()
                
                total_iteration = epoch * max_iteration + iteration               
                if total_iteration % checkpoint_iteration == 0:
                    self.save_parameters()
                    loss_validation = self.CheckErrorAndLoss(dataReader, batch_x, batch_y, epoch, total_iteration)
                    if loss_validation is not None and loss_validation <= self.params.eps:
                        break                
                #end if
            # end for
            #self.save_parameters()
            dataReader.Shuffle()
            if loss_validation is not None and loss_validation <= self.params.eps:
                break
            # end if
        # end for

        t1 = time.time()
        print("time used:", t1 - t0)

        if need_test:
            print("testing...")
            c,n = self.Test(dataReader)
            print(str.format("rate={0} / {1} = {2}", c, n, c / n))
        # end if

    def CheckErrorAndLoss(self, dataReader, train_x, train_y, epoch, total_iteration):
        print("epoch=%d, total_iteration=%d" %(epoch, total_iteration))
        # calculate train loss
        loss_train = self.lossFunc.CheckLoss(train_y, self.output)
        if self.params.loss_func_name == LossFunctionName.MSE:
            accuracy_train = self.params.eps / loss_train
        elif self.params.loss_func_name == LossFunctionName.CrossEntropy3:
            accuracy_train = self.__CalAccuracy(self.output, train_y, 3) / train_y.shape[1]
        elif self.params.loss_func_name == LossFunctionName.CrossEntropy2:
            accuracy_train = self.__CalAccuracy(self.output, train_y, 2) / train_y.shape[1]
        print("loss_train=%.4f, accuracy_train=%f" %(loss_train, accuracy_train))
        # calculate validation loss
        vld_x, vld_y = dataReader.GetDevSet()
        if vld_x is None or vld_y is None:
            self.loss_history.Add(epoch, total_iteration, loss_train, accuracy_train, None, None)
            return None
        else:
            self.__forward(vld_x, train=False)
            loss_vld = self.lossFunc.CheckLoss(vld_y, self.output)
            if self.params.loss_func_name == LossFunctionName.MSE:
                accuracy_vld = self.params.eps / loss_vld
            elif self.params.loss_func_name == LossFunctionName.CrossEntropy2:
                accuracy_vld = self.__CalAccuracy(self.output, vld_y, 2) / vld_y.shape[1]
            elif self.params.loss_func_name == LossFunctionName.CrossEntropy3:
                accuracy_vld = self.__CalAccuracy(self.output, vld_y, 3) / vld_y.shape[1]
            print("loss_valid=%.4f, accuracy_valid=%f" %(loss_vld, accuracy_vld))
            self.loss_history.Add(epoch, total_iteration, loss_train, accuracy_train, loss_vld, accuracy_vld)
            return loss_vld
        # end if
        
    def Test(self, dataReader):
        correct = 0
        test_batch = 1000
        max_iteration = max(dataReader.num_test//test_batch,1)
        for i in range(max_iteration):
            x, y = dataReader.GetBatchTestSamples(test_batch, i)
            self.__forward(x, train=False)
            correct += self.__CalAccuracy(self.output, y, 3)
        #end for
        return correct, dataReader.num_test

    # mode: 1=fitting, 2=binary classifier, 3=multiple classifier
    def __CalAccuracy(self, a, y, mode):
        if mode == 1:
            pass
        elif mode == 2:
            a[a>0]=1
            r = (a == y)
            correct = r.sum()
            return correct
        else:
            ra = np.argmax(a, axis=0)
            ry = np.argmax(y, axis=0)
            r = (ra == ry)
            correct = r.sum()
            return correct


    def inference(self, X):
        self.__forward(X)
        return self.output

    # save weights value when got low loss than before
    def save_parameters(self):
        for i in range(self.layer_count):
            layer = self.layer_list[i]
            name = self.layer_name[i]
            layer.save_parameters(name)

    # load weights for the most low loss moment
    def load_parameters(self):
        for i in range(self.layer_count):
            layer = self.layer_list[i]
            name = self.layer_name[i]
            layer.load_parameters(name)

    def ShowLossHistory(self, xmin=None, xmax=None, ymin=None, ymax=None):
        self.loss_history.ShowLossHistory(self.params, xmin, xmax, ymin, ymax)