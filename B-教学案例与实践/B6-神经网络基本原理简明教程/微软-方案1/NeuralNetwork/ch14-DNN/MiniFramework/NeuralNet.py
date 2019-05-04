# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.


import numpy as np
from enum import Enum

from MiniFramework.Layer import *
from MiniFramework.FCLayer import *
from MiniFramework.Parameters import *

class NeuralNet(object):
    def __init__(self, params):
        self.params = params
        self.layer_list = []
        self.layer_name = []
        self.output = np.zeros((1,1))
        self.layer_count = 0

    def add_layer(self, layer, name=""):
        layer.Initialize(self.params)
        self.layer_list.append(layer)
        self.layer_name.append(name)
        self.layer_count += 1

    def __forward(self, X):
        input = X
        for i in range(self.layer_count):
            layer = self.layer_list[i]
            output = layer.forward(input)
            input = output

        self.output = output
        return self.output

    def __backward(self, X, Y):
        delta_in = self.output - Y
        for i in range(self.layer_count-1,-1,-1):
            layer = self.layer_list[i]
            flag = self.__get_layer_index(i)
            delta_out = layer.backward(delta_in, flag)
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

    def train(self, dataReader, loss_history):
        loss = 0 
        lossFunc = CLossFunction(self.params.loss_func_name)
        # if num_example=200, batch_size=10, then iteration=200/10=20
        if self.params.batch_size == -1 or self.params.batch_size > dataReader.num_example:
            self.params.batch_size = dataReader.num_example
        max_iteration = (int)(dataReader.num_example / self.params.batch_size)
        for epoch in range(self.params.max_epoch):
            for iteration in range(max_iteration):
                # get x and y value for one sample
                batch_x, batch_y = dataReader.GetBatchSamples(self.params.batch_size, iteration)
                # for optimizers which need pre-update weights
                if self.params.optimizer_name == OptimizerName.Nag:
                    self.__pre_update()
                # get z from x,y
                self.__forward(batch_x)
                # calculate gradient of w and b
                self.__backward(batch_x, batch_y)
                # final update w,b
                self.__update()

                if iteration % 1000 == 0:
                    self.__forward(dataReader.X)
                    loss = lossFunc.CheckLoss(dataReader.Y, self.output)
                    print("epoch=%d, iteration=%d, loss=%f" %(epoch,iteration,loss))
                    is_min = loss_history.AddLossHistory(loss, epoch, iteration)                
                    if is_min:
                        self.save_parameters()
                #end if
            # end for            
            dataReader.Shuffle()
            if loss < self.params.eps:
                break
            # end if
            #dataReader.Shuffle()
        # end for
        
    def Test(self, dataReader):
        X = dataReader.XTestSet
        Y = dataReader.YTestSet
        correct = 0
        count = X.shape[1]
        for i in range(count):
            x = X[:,i].reshape(dataReader.num_feature, 1)
            y = Y[:,i].reshape(dataReader.num_category, 1)
            self.__forward(x)
            if np.argmax(self.output) == np.argmax(y):
                correct += 1

        return correct, count

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
