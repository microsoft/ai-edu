# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# multiple iteration, loss calculation, stop condition, predication
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SimpleDataReader import *

file_name = "../../data/ch04.npz"

class CParameters(object):
    def __init__(self, eta=0.1, max_epoch=1000, batch_size=5, eps=0.1):
        self.eta = eta
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.eps = eps

    def toString(self):
        title = str.format("bz:{0},eta:{1}", self.batch_size, self.eta)
        return title

# 帮助类，用于记录损失函数值极其对应的权重/迭代次数
class CLossHistory(object):
    def __init__(self):
        # loss history
        self.iteration = []
        self.loss_history = []
        self.w_history = []
        self.b_history = []

    def AddLossHistory(self, iteration, loss, w, b):
        self.iteration.append(iteration)
        self.loss_history.append(loss)
        self.w_history.append(w)
        self.b_history.append(b)

    # 图形显示损失函数值历史记录
    def ShowLossHistory(self, params, xmin=None, xmax=None, ymin=None, ymax=None):
        plt.plot(self.iteration, self.loss_history)
        title = params.toString()
        plt.title(title)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        if xmin != None and ymin != None:
            plt.axis([xmin, xmax, ymin, ymax])
        plt.show()
        return title

    def GetLast(self):
        count = len(self.loss_history)
        return self.loss_history[count-1], self.w_history[count-1], self.b_history[count-1]
# end class

def ForwardCalculationBatch(W,B,batch_x):
    Z = np.dot(batch_x, W) + B
    return Z

def BackPropagationBatch(batch_x, batch_y, batch_z):
    m = batch_x.shape[0]
    dZ = batch_z - batch_y
    dB = dZ.sum(axis=0, keepdims=True)/m
    dW = np.dot(batch_x.T, dZ)/m
    return dW, dB

def UpdateWeights(w, b, dW, dB, eta):
    w = w - eta*dW
    b = b - eta*dB
    return w,b

def InitialWeights(num_input, num_output, flag):
    if flag == 0:
        # zero
        W = np.zeros((num_output, num_input))
    elif flag == 1:
        # normalize
        W = np.random.normal(size=(num_output, num_input))
    elif flag == 2:
        # xavier
        W=np.random.uniform(
            -np.sqrt(6/(num_input+num_output)),
            np.sqrt(6/(num_input+num_output)),
            size=(num_output,num_input))

    B = np.zeros((num_output, 1))
    return W,B

def ShowResult(X, Y, w, b, iteration):
    # draw sample data
    plt.plot(X, Y, "b.")
    # draw predication data
    PX = np.linspace(0,1,10)
    PZ = w*PX + b
    plt.plot(PX, PZ, "r")
    plt.title("Air Conditioner Power")
    plt.xlabel("Number of Servers(K)")
    plt.ylabel("Power of Air Conditioner(KW)")
    plt.show()
    print("iteration=",iteration)
    print("w=%f,b=%f" %(w,b))

def CheckLoss(dataReader, W, B):
    X,Y = dataReader.GetWholeTrainSamples()
    m = X.shape[0]
    Z = np.dot(X, W) + B
    LOSS = (Z - Y)**2
    loss = LOSS.sum()/m/2
    return loss

def loss_contour(dataReader,loss_history):
    last_loss, result_w, result_b = loss_history.GetLast()
    X,Y=dataReader.GetWholeTrainSamples()
    w = np.linspace(result_w-1,result_w+1)
    b = np.linspace(result_b-1,result_b+1)
    W,B = np.meshgrid(w,b)
    Z = np.dot(X, W.ravel().reshape(1,2500)) + B.ravel().reshape(1,2500)
    plt.contour(X,Y,Z)

def loss_2d(dataReader,loss_history,batch_size,epoch):

    x,y=dataReader.GetWholeTrainSamples()
    n = dataReader.num_train

    last_loss, result_w, result_b = loss_history.GetLast()

    # show contour of loss
    s = 150
    W = np.linspace(result_w-1,result_w+1,s)
    B = np.linspace(result_b-1,result_b+1,s)
    LOSS = np.zeros((s,s))
    for i in range(len(W)):
        for j in range(len(B)):
            w = W[i]
            b = B[j]
            a = np.dot(x, w) + b
            loss = CheckLoss(dataReader,w,b)
            LOSS[i,j] = np.round(loss, 2)
        # end for j
    # end for i
    print("please wait for 20 seconds...")
    while(True):
        X = []
        Y = []
        is_first = True
        loss = 0
        for i in range(len(W)):
            for j in range(len(B)):
                if LOSS[i,j] != 0:
                    if is_first:
                        loss = LOSS[i,j]
                        X.append(W[i])
                        Y.append(B[j])
                        LOSS[i,j] = 0
                        is_first = False
                    elif LOSS[i,j] == loss:
                        X.append(W[i])
                        Y.append(B[j])
                        LOSS[i,j] = 0
                    # end if
                # end if
            # end for j
        # end for i
        if is_first == True:
            break
        plt.plot(X,Y,'.')
    # end while

    # show w,b trace
    w_history = loss_history.w_history
    b_history = loss_history.b_history
    plt.plot(w_history,b_history)

    plt.xlabel("w")
    plt.ylabel("b")
    title = str.format("bz={0}, Epoch={1}, Loss={2:.3f}, W={3:.3f}, B={4:.3f}", batch_size, epoch, last_loss, result_w, result_b)
    plt.title(title)
    plt.axis([result_w-1,result_w+1,result_b-1,result_b+1])
    plt.show()


def train(params):
        
    W, B = InitialWeights(1,1,0)
    # calculate loss to decide the stop condition
    loss_history = CLossHistory()
    # read data
    sdr = SimpleDataReader(file_name)
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
            if iteration % 10 == 0:
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
    #ShowResult(X, Y, W, B, epoch)
    print(W,B)

    x = 346/1000
    result = ForwardCalculationBatch(W, B, x)
    print("346 machines need the power of the air-conditioner=",result)
    
    #loss_2d(sdr,loss_history,params.batch_size,epoch)
    loss_contour(sdr, loss_history)

