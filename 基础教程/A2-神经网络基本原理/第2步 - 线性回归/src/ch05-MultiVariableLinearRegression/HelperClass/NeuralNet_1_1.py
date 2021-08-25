# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

"""
Version 1.1
what's new: 
- W,B are array instead of number
- don't show loss contour because number of parameters are larger than 2
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
from matplotlib.colors import LogNorm

from HelperClass.DataReader_1_1 import *
from HelperClass.HyperParameters_1_0 import *
from HelperClass.TrainingHistory_1_0 import *

class NeuralNet_1_1(object):
    def __init__(self, hp):
        self.hp = hp
        self.W = np.zeros((self.hp.input_size, self.hp.output_size))
        self.B = np.zeros((1, self.hp.output_size))

    def __forwardBatch(self, batch_x):
        Z = np.dot(batch_x, self.W) + self.B
        return Z

    def __backwardBatch(self, batch_x, batch_y, batch_z):
        m = batch_x.shape[0]
        dZ = batch_z - batch_y
        dB = dZ.sum(axis=0, keepdims=True)/m
        dW = np.dot(batch_x.T, dZ)/m
        return dW, dB

    def __update(self, dW, dB):
        self.W = self.W - self.hp.eta * dW
        self.B = self.B - self.hp.eta * dB

    def inference(self, x):
        return self.__forwardBatch(x)

    def train(self, dataReader, checkpoint=0.1):
        # calculate loss to decide the stop condition
        loss_history = TrainingHistory_1_0()
        loss = 10
        if self.hp.batch_size == -1:
            self.hp.batch_size = dataReader.num_train
        max_iteration = math.ceil(dataReader.num_train / self.hp.batch_size)
        checkpoint_iteration = (int)(max_iteration * checkpoint)

        for epoch in range(self.hp.max_epoch):
            print("epoch=%d" %epoch)
            dataReader.Shuffle()
            for iteration in range(max_iteration):
                # get x and y value for one sample
                batch_x, batch_y = dataReader.GetBatchTrainSamples(self.hp.batch_size, iteration)
                # get z from x,y
                batch_z = self.__forwardBatch(batch_x)
                # calculate gradient of w and b
                dW, dB = self.__backwardBatch(batch_x, batch_y, batch_z)
                # update w,b
                self.__update(dW, dB)

                total_iteration = epoch * max_iteration + iteration
                if (total_iteration+1) % checkpoint_iteration == 0:
                    loss = self.__checkLoss(dataReader)
                    print(epoch, iteration, loss, self.W, self.B)
                    loss_history.AddLossHistory(epoch*max_iteration+iteration, loss)
                    if loss < self.hp.eps:
                        break
                    #end if
                #end if
            # end for
            if loss < self.hp.eps:
                break
        # end for
        loss_history.ShowLossHistory(self.hp)
        print("W=", self.W)
        print("B=", self.B)

    def __checkLoss(self, dataReader):
        X,Y = dataReader.GetWholeTrainSamples()
        m = X.shape[0]
        Z = self.__forwardBatch(X)
        LOSS = (Z - Y)**2
        loss = LOSS.sum()/m/2
        return loss



