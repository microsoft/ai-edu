# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# multiple iteration, loss calculation, stop condition, predication
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LogNorm

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

def loss_contour(dataReader,loss_history,batch_size,iteration):
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
    show_wb_history(loss_history, result_w, result_b, batch_size, iteration)
    plt.show()


def show_wb_history(loss_history, result_w, result_b, batch_size, iteration):
    # show w,b trace
    w_history = loss_history.w_history
    b_history = loss_history.b_history
    plt.plot(w_history,b_history)

    plt.xlabel("w")
    plt.ylabel("b")
    title = str.format("batchsize={0}, iteration={1}, w={2:.3f}, b={3:.3f}", batch_size, iteration, result_w, result_b)
    plt.title(title)
    plt.axis([result_w-1,result_w+1,result_b-1,result_b+1])


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
    
    loss_contour(sdr, loss_history, params.batch_size, epoch*max_iteration+iteration)

