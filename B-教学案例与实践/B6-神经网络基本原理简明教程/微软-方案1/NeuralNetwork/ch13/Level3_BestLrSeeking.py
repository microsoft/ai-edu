# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math

from LossFunction import * 
from Parameters import *
from Activators import *
from DataReader import *
from Level0_TwoLayerNet import *

x_data_name = "X8.dat"
y_data_name = "Y8.dat"

class Looper(object):
    # e.g. (0.01, 0.005, 100) means start from 0.01, every 100 iteration will add 0.005 until 0.01*10 (=0.1)
    # in another word: 0.01,0.015,0.02,0.025,...0.09,0.095,0.1 (stop)
    def __init__(self, start, step, loop, stop=-1):
        self.start = start
        if stop == -1:
            self.stop = start * 10
        else:
            self.stop = stop
        self.step = step
        self.loop = loop

    # 把Looper展开成两个一维数组,后面便于计算
    def to_array(self):
        lrs = []
        loops = []
        i = 0
        while(True):
            lr = self.start + i * self.step
            if lr > self.stop:
                break
            else:
                lrs.append(lr)
                loops.append(self.loop)
                i = i + 1
            # end if
        #end while
        return lrs, loops

class LrLooper(object):
    def __init__(self):
        self.learningRates = []
        self.loopCount = []
        self.searchIndex = -1
        
        # 记录训练过程中当前的lr对应的loss值
        self.loss_history = []
        self.lr_history = []

    def addLooper(self,looper):
        self.searchIndex = 0
        lrs, loops = looper.to_array()
        self.learningRates.extend(lrs)
        self.loopCount.extend(loops)

    def getFirstLearningRate(self):
        self.searchIndex = 0
        lr = self.learningRates[self.searchIndex]
        loop = self.loopCount[self.searchIndex]
        self.searchIndex = self.searchIndex + 1
        return lr, loop

    def getNextLearningRate(self):
        if self.searchIndex == -1 or self.searchIndex >= len(self.learningRates):
            return None,None
        #end if
        
        lr = self.learningRates[self.searchIndex]
        loop = self.loopCount[self.searchIndex]
        self.searchIndex = self.searchIndex + 1
        return lr, loop

    def addHistory(self, loss, lr):
        self.loss_history.append(loss)
        self.lr_history.append(lr)

    def getLrLossHistory(self):
        return self.lr_history, self.loss_history


class LrSeekingNet(TwoLayerNet):

    def UpdateWeights(self, wb1, wb2, lr):
        wb1.UpdateWithLr(lr)
        wb2.UpdateWithLr(lr)

    def train(self, dataReader, params, loss_history, lr_searcher):
        # initialize weights and bias
        wb1 = WeightsBias(params.num_input, params.num_hidden, params.eta, params.init_method)
        wb1.InitializeWeights(False)
        wb2 = WeightsBias(params.num_hidden, params.num_output, params.eta, params.init_method)
        wb2.InitializeWeights(False)
        # calculate loss to decide the stop condition
        loss = 0 
        lossFunc = CLossFunction(self.loss_func_name)

        lr, loop = lr_searcher.getFirstLearningRate()         # 特意增加的逻辑代码

        # if num_example=200, batch_size=10, then iteration=200/10=20
        max_iteration = (int)(dataReader.num_example / params.batch_size)
        for epoch in range(params.max_epoch):
            #dataReader.Shuffle()
            for iteration in range(max_iteration):
                # get x and y value for one sample
                batch_x, batch_y = dataReader.GetBatchSamples(params.batch_size,iteration)
                # get z from x,y
                dict_cache = self.forward(batch_x, wb1, wb2)
                # calculate gradient of w and b
                self.BackPropagationBatch(batch_x, batch_y, dict_cache, wb1, wb2)
                # update w,b
                self.UpdateWeights(wb1, wb2, lr) # 特意修改的逻辑代码
            # end for            
            # calculate loss for this batch
            output = self.forward(dataReader.X, wb1, wb2)
            loss = lossFunc.CheckLoss(dataReader.Y, output["Output"])
            print("epoch=%d, loss=%f, lr=%f" %(epoch,loss,lr))
            loss_history.AddLossHistory(loss, epoch, iteration, wb1, wb2)          
            
            # 特意增加的逻辑代码
            lr_searcher.addHistory(loss, lr)
            if (epoch+1) % loop == 0:   # avoid when epoch==0
                lr, loop = lr_searcher.getNextLearningRate()
            if lr == None:
                break
        # end for
        return wb1, wb2
    # end def

def try_1():
    # try 1    
    lr_Searcher = LrLooper()
    looper = Looper(0.0001,0.0001,50)
    lr_Searcher.addLooper(looper)
    looper = Looper(0.001,0.001,50)
    lr_Searcher.addLooper(looper)
    looper = Looper(0.01,0.01,50)
    lr_Searcher.addLooper(looper)
    looper = Looper(0.1,0.1,50)
    lr_Searcher.addLooper(looper) 
    looper = Looper(1,0.01,50,1.2)
    lr_Searcher.addLooper(looper) 
    return lr_Searcher

def try_2():
    # try 2
    lr_Searcher = LrLooper()
    looper = Looper(0.5,0.01,50,1.0)
    lr_Searcher.addLooper(looper)
    return lr_Searcher

def try_3():
    lr_Searcher = LrLooper()
    looper = Looper(0.8,0.01,200,1.1)
    lr_Searcher.addLooper(looper)
    return lr_Searcher


def ShowResult(net, X, Y, title, wb1, wb2):
    # draw train data
    plt.plot(X[0,:], Y[0,:], '.', c='b')
    # create and draw visualized validation data
    TX = np.linspace(0,1,100).reshape(1,100)
    dict_cache = net.forward(TX, wb1, wb2)
    TY = dict_cache["Output"]
    plt.plot(TX, TY, 'x', c='r')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    dataReader = DataReader(x_data_name, y_data_name)
    XData,YData = dataReader.ReadData()
    X = dataReader.NormalizeX(passthrough=True)
    Y = dataReader.NormalizeY()
    # 为了说明问题，我们用2个隐层单元和20批大小来做试验
    n_input, n_hidden, n_output = 1, 2, 1
    eta, batch_size, max_epoch = 0.1, 20, 30000
    eps = 0.001

    params = CParameters(n_input, n_hidden, n_output,
                         eta, max_epoch, batch_size, eps, 
                         InitialMethod.Xavier,
                         OptimizerName.SGD)

    loss_history = CLossHistory()

    # 试验时，先注释掉try_2，运行try_1；然后注释掉try_1，运行try_2
    lr_Searcher = try_1()
    # lr_Searcher = try_2()


    net = LrSeekingNet(NetType.Fitting)
    wb1, wb2 = net.train(dataReader, params, loss_history, lr_Searcher)

    lrs, losses = lr_Searcher.getLrLossHistory()
    plt.plot(np.log10(lrs), losses)
    plt.show()
    ShowResult(net, X, Y, "", wb1, wb2)



