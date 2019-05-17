# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.


import numpy as np

from MiniFramework.Layer import *
from MiniFramework.FCLayer import *
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

    # checkpoint=0.1 means will calculate the loss/accuracy every 10% in each epoch
    def train(self, dataReader, loss_history, checkpoint=0.1):
        loss = 0 
        lossFunc = CLossFunction(self.params.loss_func_name)
        # if num_example=200, batch_size=10, then iteration=200/10=20
        if self.params.batch_size == -1 or self.params.batch_size > dataReader.num_example:
            self.params.batch_size = dataReader.num_train
        max_iteration = dataReader.num_train // self.params.batch_size
        checkpoint_iteration = max_iteration * checkpoint
        for epoch in range(self.params.max_epoch):
            for iteration in range(max_iteration):
                # get x and y value for one sample
                batch_x, batch_y = dataReader.GetBatchTrainSamples(self.params.batch_size, iteration)
                # for optimizers which need pre-update weights
                if self.params.optimizer_name == OptimizerName.Nag:
                    self.__pre_update()
                # get z from x,y
                self.__forward(batch_x)
                # calculate gradient of w and b
                self.__backward(batch_x, batch_y)
                # final update w,b
                self.__update()

                if iteration % checkpoint_iteration == 0:
                    self.CheckErrorAndLoss(dataReader, lossFunc, batch_x, batch_y, loss_history, epoch, iteration*self.params.batch_size)
                #end if
            # end for    
            self.save_parameters()
            dataReader.Shuffle()
            # end if
        # end for
        print("testing...")
        c,n = self.Test(dataReader)
        print(str.format("rate={0} / {1} = {2}", c, n, c / n))


    def CheckErrorAndLoss(self, dataReader, lossFunc, batch_x, batch_y, loss_history, epoch, iteration):
        loss_train = lossFunc.CheckLoss(batch_y, self.output)
        accuracy_train = self.__CalAccuracy(self.output, batch_y) / batch_y.shape[1]
    
        self.__forward(dataReader.XDevSet)
        loss_val = lossFunc.CheckLoss(dataReader.YDevSet.T, self.output)
        y = dataReader.YDevSet.T
        accuracy_val = self.__CalAccuracy(self.output, y) / y.shape[1]
    
        loss_history.Add(epoch, iteration, loss_train, accuracy_train, loss_val, accuracy_val)
        print("epoch=%d, iteration=%d" %(epoch, iteration))
        print("loss_train=%.3f, accuracy_train=%f" %(loss_train, accuracy_train))
        print("loss_valid=%.3f, accuracy_valid=%f" %(loss_val, accuracy_val))

        return

    def Test(self, dataReader):
        correct = 0
        test_batch = 1000
        max_iteration = dataReader.num_test//test_batch
        for i in range(max_iteration):
            x, y = dataReader.GetBatchTestSamples(test_batch, i)
            self.__forward(x)
            correct += self.__CalAccuracy(self.output, y)
        #end for
        return correct, dataReader.num_test

    def __CalAccuracy(self, a, y_onehot):
        ra = np.argmax(a, axis=0).reshape(-1,1)
        ry = np.argmax(y_onehot, axis=0).reshape(-1,1)
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
