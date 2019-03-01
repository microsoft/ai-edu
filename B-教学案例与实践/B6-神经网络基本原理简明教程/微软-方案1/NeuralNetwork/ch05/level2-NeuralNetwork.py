# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# warning: 运行本程序将会得到失败的结果，这是by design的，是为了讲解课程内容，后面的程序中会有补救的方法

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class CData(object):
    def __init__(self, loss, w, b, epoch, iteraion):
        self.loss = loss
        self.w = w
        self.b = b
        self.epoch = epoch
        self.iteration = iteration


# 加载数据
def ReadData():
    Xfile = Path("HousePriceXData.dat")
    Yfile = Path("HousePriceYData.dat")
    if Xfile.exists() & Yfile.exists():
        XData = np.load(Xfile)
        YData = np.load(Yfile)
        return XData,YData
    
    return None,None

# 前向计算
def ForwardCalculationBatch(W, B, batch_X):
    Z = np.dot(W, batch_X) + B
    return Z

# X:input example, Y:lable example, Z:predicated value
def BackPropagationBatch(batch_X, batch_Y, Z):
    m = batch_X.shape[1]
    dZ = Z - batch_Y
    # dZ列相加，即一行内的所有元素相加
    dB = dZ.sum(axis=1, keepdims=True)/m
    dW = np.dot(dZ, batch_X.T)/m
    return dW, dB

# 更新权重参数
def UpdateWeights(W, B, dW, dB, eta):
    W = W - eta*dW
    B = B - eta*dB
    return W, B

# 计算损失函数值
def CheckLoss(W, B, X, Y):
    m = X.shape[1]
    Z = ForwardCalculationBatch(W,B,X)
    LOSS = (Z - Y)**2
    loss = LOSS.sum()/m/2
    return loss

def GetBatchSamples(X,Y,batch_size,iteration):
    num_feature = X.shape[0]
    start = iteration * batch_size
    end = start + batch_size
    batch_X = X[0:num_feature, start:end].reshape(num_feature, batch_size)
    batch_Y = Y[0, start:end].reshape(1, batch_size)
    return batch_X, batch_Y

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

def InitializeHyperParameters(method):
    if method=="SGD":
        eta = 0.1
        max_epoch = 1
        batch_size = 1
    elif method=="MiniBatch":
        eta = 0.1
        max_epoch = 50
        batch_size = 10
    elif method=="FullBatch":
        eta = 0.5
        max_epoch = 100
        batch_size = 200
    return eta, max_epoch, batch_size

def GetMinimalLossData(dict_loss):
    key = sorted(dict_loss.keys())[0]
    w = dict_loss[key].w
    b = dict_loss[key].b
    return w,b,dict_loss[key]

def ShowLossHistory(dict_loss, method):
    loss = []
    for key in dict_loss:
        loss.append(key)

    #plt.plot(loss)
    plt.plot(loss[30:800])
    plt.title(method)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()



if __name__ == '__main__':
    # hyper parameters
    # SGD, MiniBatch, FullBatch
    method = "SGD"

    eta, max_epoch,batch_size = InitializeHyperParameters(method)
    
    # W size is 3x1, B is 1x1
    W, B = InitialWeights(3,1,2)
    # calculate loss to decide the stop condition
    loss = 5
    dict_loss = {}
    # read data
    X, Y = ReadData()
    # count of samples
    num_example = X.shape[1]
    num_feature = X.shape[0]


    # if num_example=200, batch_size=10, then iteration=200/10=20
    max_iteration = (int)(num_example / batch_size)
    for epoch in range(max_epoch):
        print("epoch=%d" %epoch)
        for iteration in range(max_iteration):
            # get x and y value for one sample
            batch_x, batch_y = GetBatchSamples(X,Y,batch_size,iteration)
            # get z from x,y
            batch_z = ForwardCalculationBatch(W, B, batch_x)
            # calculate gradient of w and b
            dW, dB = BackPropagationBatch(batch_x, batch_y, batch_z)
            # update w,b
            W, B = UpdateWeights(W, B, dW, dB, eta)
            
            # calculate loss for this batch
            loss = CheckLoss(W,B,X,Y)
            print(epoch,iteration,loss,W,B)
            prev_loss = loss

            dict_loss[loss] = CData(loss, W, B, epoch, iteration)            

    ShowLossHistory(dict_loss, method)
    w,b,cdata = GetMinimalLossData(dict_loss)
    print(cdata.w, cdata.b)
    print("epoch=%d, iteration=%d, loss=%f" %(cdata.epoch, cdata.iteration, cdata.loss))




'''
flag = 'x_only'
XData, YData = ReadData()
X, X_range, X_min, Y, Y_range, Y_min = NormalizeXY(XData, YData, flag)

m = X.shape[1]  # count of examples
n = X.shape[0]  # feature count
eta = 0.1   # learning rate
loss, diff_loss, prev_loss = 10, 10, 5
eps = 1e-10
max_iteration = 100 # 最多100次循环
# 初始化w,b
B = np.zeros((1,1))
W = np.zeros((1,n))

for iteration in range(max_iteration):
    for i in range(m):
        Xm = X[0:n,i].reshape(n,1)
        Ym = Y[0,i].reshape(1,1)
        Z = ForwardCalculation(Xm, W, B)
        dw, db = BackPropagation(Xm, Ym, Z)
        W, B = UpdateWeights(W, B, dw, db, eta)
        
        loss, diff_loss = CheckLoss(W,B,X,Y,m,prev_loss)
        if i%10==0:
            print(iteration, i, loss, diff_loss)
        if diff_loss < eps:
            break
        prev_loss = loss
    if diff_loss < eps:
        break

print("W=", W)
print("B=", B)
W_real, B_real = DeNormalizeWeights(X_range, XData, n)
PredicateTest(W_real, B_real, W, B, flag)
'''
