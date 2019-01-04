# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path

def ReadData():
    Xfile = Path("HousePriceXData.dat")
    Yfile = Path("HousePriceYData.dat")
    if Xfile.exists() & Yfile.exists():
        XData = np.load(Xfile)
        YData = np.load(Yfile)
        return XData,YData
    
    return None,None

def ForwardCalculation(Xm,W,b):
    z = np.dot(W, Xm) + b
    return z

def CheckLoss(w, b, X, Y, count, prev_loss):
    Z = ForwardCalculation(X, w, b)
    LOSS = (Z - Y)**2
    loss = LOSS.sum()/count/2
    diff_loss = abs(loss - prev_loss)
    return loss, diff_loss

def BackPropagation(Xm,Y,Z):
    dloss_z = Z - Y
    db = dloss_z
    dw = np.dot(dloss_z, Xm.T)
    return dw, db

def UpdateWeights(w, b, dW, dB, eta):
    w = w - eta*dW
    b = b - eta*dB
    return w,b

# normalize data by extracting range from source data
def NormalizeByData(X):
    X_new = np.zeros(X.shape)
    n = X.shape[0]
    x_range = np.zeros((1,n))
    x_min = np.zeros((1,n))
    for i in range(n):
        x = X[i,:]
        max_value = np.max(x)
        min_value = np.min(x)
        x_min[0,i] = min_value
        x_range[0,i] = max_value - min_value
        x_new = (x - x_min[0,i])/(x_range[0,i])
        X_new[i,:] = x_new
    return X_new, x_range, x_min

# normalize data by specified range and min_value
def NormalizeByRange(X, x_range, x_min):
    X_new = np.zeros(X.shape)
    n = X.shape[0]
    for i in range(n):
        x = X[i,:]
        x_new = (x-x_min[0,i])/x_range[0,i]
        X_new[i,:] = x_new
    return X_new

def NormalizeXY(XData, YData, flag):
    if flag=='x_only':
        X, X_range, X_min = NormalizeByData(XData)
        return X, X_range, X_min, YData, -1, -1
    elif flag=='x_and_y':
        X, X_range, X_min = NormalizeByData(XData)
        Y, Y_range, Y_min = NormalizeByData(YData)
        return X, X_range, X_min, Y, Y_range, Y_min

# get real weights
def DeNormalizeWeights(X_range, XData, n):
    W_real = np.zeros((1,n))
    for i in range(n):
        W_real[0,i] = W[0,i] / X_range[0,i]
    print("W_real=", W_real)

    B_real = 0
    num_sample = XData.shape[1]
    for i in range(num_sample):
        xm = XData[0:n,i].reshape(n,1)
        zm = ForwardCalculation(xm, W_real, 0)
        ym = Y[0,i].reshape(1,1)
        B_real = B_real + (ym - zm)
    B_real = B_real / num_sample
    print("B_real=", B_real)
    return W_real, B_real

# try to give the answer for the price of 朝西(2)，五环(5)，93平米的房子
def PredicateTest(W_real, B_real, W, B, flag):
    xt = np.array([2,5,93]).reshape(3,1)
    z1 = ForwardCalculation(xt, W_real, B_real)

    xt_new = NormalizeByRange(xt, X_range, X_min)
    z2 = ForwardCalculation(xt_new, W, B)

    if flag == 'x_only':
        print("xt,W_real,B_real:",z1)
        print("xt_new,W,B:",z2)
    elif flag == 'x_and_y':
        print("xt,W_real,B_real:", z1*Y_range+Y_min)
        print("xt_new,W,B:", z2*Y_range+Y_min)
    

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




