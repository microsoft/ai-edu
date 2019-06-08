# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# multiple iteration, loss calculation, stop condition, predication
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LogNorm

from HelperClass.SimpleDataReader import *
from HelperClass.HyperParameters import *
from HelperClass.TrainingHistory import *

class NeuralNet(object):
    def __init__(self, params):
        self.params = params
        self.W = np.zeros((1,1))
        self.B = np.zeros((1,1))

    def ForwardCalculationBatch(self, batch_x):
        Z = np.dot(batch_x, self.W) + self.B
        return Z

    def BackPropagationBatch(self, batch_x, batch_y, batch_z):
        m = batch_x.shape[0]
        dZ = batch_z - batch_y
        dB = dZ.sum(axis=0, keepdims=True)/m
        dW = np.dot(batch_x.T, dZ)/m
        return dW, dB

    def UpdateWeights(self, dW, dB):
        self.W = self.W - self.params.eta * dW
        self.B = self.B - self.params.eta * dB

    def train(self, dataReader):
        # calculate loss to decide the stop condition
        loss_history = TrainingHistory()
        # read data
        sdr = SimpleDataReader()
        sdr.ReadData()

        if params.batch_size == -1:
            params.batch_size = sdr.num_train
        max_iteration = (int)(sdr.num_train / params.batch_size)
        for epoch in range(params.max_epoch):
            print("epoch=%d" %epoch)
            sdr.Shuffle()
            for iteration in range(max_iteration):
                # get x and y value for one sample
                batch_x, batch_y = sdr.GetBatchTrainSamples(params.batch_size, iteration)
                # get z from x,y
                batch_z = ForwardCalculationBatch(W, B, batch_x)
                # calculate gradient of w and b
                dW, dB = BackPropagationBatch(batch_x, batch_y, batch_z)
                # update w,b
                W, B = UpdateWeights(W, B, dW, dB, params.eta)
                if iteration % 2 == 0:
                    loss = CheckLoss(sdr,W,B)
                    print(epoch, iteration, loss, W, B)
                    loss_history.AddLossHistory(epoch*max_iteration+iteration, loss, W[0,0], B[0,0])
                    if loss < params.eps:
                        break
                    #end if
                #end if
                if loss < params.eps:
                    break
            # end for
            if loss < params.eps:
                break
        # end for
        loss_history.ShowLossHistory(params)
        print(W,B)

        x = 346/1000
        result = ForwardCalculationBatch(W, B, x)
        print("346 machines need the power of the air-conditioner=",result)
    
        ShowResult(sdr, W, B, loss)
        loss_contour(sdr, loss_history, params.batch_size, epoch*max_iteration+iteration)

    def CheckLoss(self, dataReader):
        X,Y = dataReader.GetWholeTrainSamples()
        m = X.shape[0]
        Z = np.dot(X, W) + B
        LOSS = (Z - Y)**2
        loss = LOSS.sum()/m/2
        return loss

    def ShowResult(self, dataReader, w, b, loss):
        X,Y = dataReader.GetWholeTrainSamples()
        # draw sample data
        plt.plot(X, Y, "b.")
        # draw predication data
        PX = np.array([0,1]).reshape(2,1)
        PZ = w*PX + b
        plt.plot(PX, PZ, "r")
        plt.title(str.format("w={0}, b={1}, loss={2:.5f}", w, b, loss))
        plt.xlabel("Number of Servers(K)")
        plt.ylabel("Power of Air Conditioner(KW)")
        plt.show()
        print("w=%f,b=%f" %(w,b))

    def loss_contour(self, dataReader,loss_history,batch_size,iteration):
        last_loss, result_w, result_b = loss_history.GetLast()
        X,Y=dataReader.GetWholeTrainSamples()
        len1 = 50
        len2 = 50
        w = np.linspace(result_w-1,result_w+1,len1)
        b = np.linspace(result_b-1,result_b+1,len2)
        W,B = np.meshgrid(w,b)
        len = len1 * len2
        X,Y = dataReader.GetWholeTrainSamples()
        m = X.shape[0]
        Z = np.dot(X, W.ravel().reshape(1,len)) + B.ravel().reshape(1,len)
        Loss1 = (Z - Y)**2
        Loss2 = Loss1.sum(axis=0,keepdims=True)/m
        Loss3 = Loss2.reshape(len1, len2)
        plt.contour(W,B,Loss3,levels=np.logspace(-5, 5, 100), norm=LogNorm(), cmap=plt.cm.jet)

        # show w,b trace
        w_history = loss_history.w_history
        b_history = loss_history.b_history
        plt.plot(w_history,b_history)
        plt.xlabel("w")
        plt.ylabel("b")
        title = str.format("batchsize={0}, iteration={1}, w={2:.3f}, b={3:.3f}", batch_size, iteration, result_w, result_b)
        plt.title(title)

        plt.axis([result_w-1,result_w+1,result_b-1,result_b+1])
        plt.show()


