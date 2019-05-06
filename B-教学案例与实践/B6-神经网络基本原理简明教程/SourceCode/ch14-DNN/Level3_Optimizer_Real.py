# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# multiple iteration, loss calculation, stop condition, predication
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LogNorm

from MiniFramework.Optimizer import *
from MiniFramework.Parameters import *
from MiniFramework.WeightsBias import *

x_data_name = "X04.dat"
y_data_name = "Y04.dat"

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
        return self.loss_history[count-1], self.w_history[count-1], self.b_history[count-1], self.iteration[count-1]
# end class

def ReadData():
    Xfile = Path(x_data_name)
    Yfile = Path(y_data_name)
    if Xfile.exists() & Yfile.exists():
        X = np.load(Xfile)
        Y = np.load(Yfile)
        return X.reshape(1,-1),Y.reshape(1,-1)
    else:
        return None,None

def GetBatchSamples(X,Y,batch_size,iteration):
    num_feature = X.shape[0]
    start = iteration * batch_size
    end = start + batch_size
    batch_x = X[0:num_feature,start:end].reshape(num_feature,batch_size)
    batch_y = Y[0,start:end].reshape(1,batch_size)
    return batch_x, batch_y

def Shuffle(X, Y):
    seed = np.random.randint(0,100)
    np.random.seed(seed)
    XP = np.random.permutation(X.T)
    np.random.seed(seed)
    YP = np.random.permutation(Y.T)
    return XP.T, YP.T

def ForwardCalculationBatch(W,B,batch_x):
    Z = np.dot(W, batch_x) + B
    return Z

def BackPropagationBatch(batch_x, batch_y, batch_z):
    m = batch_x.shape[1]
    dZ = batch_z - batch_y
    dB = dZ.sum(axis=1, keepdims=True)/m
    dW = np.dot(dZ, batch_x.T)/m
    return dW, dB

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

def CheckLoss(W, B, X, Y):
    m = X.shape[1]
    Z = np.dot(W, X) + B
    LOSS = (Z - Y)**2
    loss = LOSS.sum()/m/2
    return loss

def show_contour(ax, loss_history, optimizer):
    # draw w,b training history
    ax.plot(loss_history.w_history, loss_history.b_history)
    # read example data
    X,Y = ReadData()
    # generate w,b data grid array for 3D, w = x_axis, b = y_axis
    w = np.arange(1, 3, 0.01)
    b = np.arange(2, 4, 0.01)
    W, B = np.meshgrid(w, b) 
    m = X.shape[1]
    # calculate Z (z_axis)
    Z = np.zeros((W.shape))
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            w = W[i,j]
            b = B[i,j]
            z = np.dot(w, X) + b
            LOSS = (z - Y)**2
            loss = LOSS.sum()/m/2
            Z[i,j] = loss
        #end for
    #end for
    # draw contour
    c = ax.contour(W, B, Z, levels=np.logspace(-4, 4, 35), norm=LogNorm(), cmap=plt.cm.jet)
    #ax.clabel(c,fontsize=6,colors=('k','r'))
    # set the drawing rectangle area
    ax.axis([1, 3, 2, 4])
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    l,w,b,i = loss_history.GetLast()
    ax.set_title(str.format("{0} loss={1:.4f} w={2:.2f} b={3:.3f} ite={4}", optimizer, l, w, b, i))


def train(params):    

    wb = WeightsBias(1,1,InitialMethod.Zero, params.optimizer_name, params.eta)
    wb.InitializeWeights()

    # calculate loss to decide the stop condition
    loss_history = CLossHistory()
    # read data
    X, Y = ReadData()
    # count of samples
    num_example = X.shape[1]
    num_feature = X.shape[0]

    # if num_example=200, batch_size=10, then iteration=200/10=20
    max_iteration = (int)(num_example / params.batch_size)
    for epoch in range(params.max_epoch):
        print("epoch=%d" %epoch)
        X,Y = Shuffle(X,Y)
        for iteration in range(max_iteration):
            # get x and y value for one sample
            batch_x, batch_y = GetBatchSamples(X,Y,params.batch_size,iteration)
            # get z from x,y
            batch_z = ForwardCalculationBatch(wb.W, wb.B, batch_x)
            # calculate gradient of w and b
            wb.dW, wb.dB = BackPropagationBatch(batch_x, batch_y, batch_z)
            # update w,b
            wb.Update()
            loss = CheckLoss(wb.W, wb.B, X, Y)
            print(epoch, iteration, loss, wb.W, wb.B)
            loss_history.AddLossHistory(epoch*max_iteration+iteration, loss, wb.W[0,0], wb.B[0,0])
            if loss < params.eps:
                break
            #end if
            if loss < params.eps:
                break
        # end for
        if loss < params.eps:
            break
    # end for
    return loss_history

if __name__ == '__main__':
    
    params = CParameters(eta=0.2, max_epoch=5, batch_size=5, eps = 0.005, optimizerName=OptimizerName.SGD)
    loss_history = train(params)
    ax = plt.subplot(2, 2, 1)
    show_contour(ax, loss_history, params.optimizer_name.name)

    params = CParameters(eta=0.02, max_epoch=5, batch_size=5, eps = 0.005, optimizerName=OptimizerName.Momentum)
    loss_history =  train(params)
    ax = plt.subplot(2, 2, 2)
    show_contour(ax, loss_history, params.optimizer_name.name)
    
    params = CParameters(eta=0.05, max_epoch=5, batch_size=5, eps = 0.005, optimizerName=OptimizerName.RMSProp)
    loss_history = train(params)
    ax = plt.subplot(2, 2, 3)
    show_contour(ax, loss_history, params.optimizer_name.name)

    params = CParameters(eta=0.2, max_epoch=5, batch_size=5, eps = 0.005, optimizerName=OptimizerName.Adam)
    loss_history = train(params)
    ax = plt.subplot(2, 2, 4)
    show_contour(ax, loss_history, params.optimizer_name.name)

    plt.suptitle("Optimizers")
    plt.show()
