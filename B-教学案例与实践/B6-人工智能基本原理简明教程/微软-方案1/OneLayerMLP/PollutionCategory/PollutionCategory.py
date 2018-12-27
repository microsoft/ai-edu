# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

x_data_name = "PollutionCategoryXData.dat"
y_data_name = "PollutionCategoryYData.dat"

def LoadData():
    Xfile = Path(x_data_name)
    Yfile = Path(y_data_name)
    if Xfile.exists() & Yfile.exists():
        XData = np.load(Xfile)
        YData = np.load(Yfile)
        return XData,YData
    
    return None,None

def ForwardCalculation(W, X, B):
    Z = np.dot(W,X) + B
    return Z

def Softmax(Z):
    shift_z = Z - np.max(Z)
    exp_z = np.exp(shift_z)
    A = exp_z / np.sum(exp_z, axis=0)
    return A

def BackPropagation(Xm, Ym, A):
    dZ = A - Ym
    dB = dZ
    dW = np.dot(dZ, Xm.T)
    return dW, dB

def UpdateWeights(W, B, dW, dB, eta):
    W = W - eta*dW
    B = B - eta*dB
    return W,B

def CheckLoss(W, B, X, Y, count, prev_loss):
    Z = ForwardCalculation(W,X,B)
    A = Softmax(Z)
    # binary classification
    #p1 = (1-Y) * np.log(1-A)  
    p2 = Y * np.log(A)
    # binary classification
    #LOSS = np.sum(-(p1 + p2))
    # multiple classification
    LOSS = np.sum(-p2)
    loss = LOSS / count
    diff_loss = abs(loss - prev_loss)
    return loss, diff_loss

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


def InitialWeights(num_category, num_features, flag):
    if flag == 'zero':
        W = np.zeros((num_category,num_features))
        B = np.zeros((num_category,1))
    elif flag == 'random':
        W = np.random.random((num_category,num_features))
        B = np.random.random((num_category,1))
    return W, B

def ShowResult(X,Y,W,B,rangeX,eta,iteration,eps):
    for i in range(X.shape[1]):
        if Y[0,i] == 0 and Y[1,i]==0 and Y[2,i]==1:
            plt.scatter(X[0,i], X[1,i], c='r')
        elif Y[0,i] == 0 and Y[1,i]==1 and Y[2,i]==0:
            plt.scatter(X[0,i], X[1,i], c='b')
        else:
            plt.scatter(X[0,i], X[1,i], c='g')
   
    b12 = (B[1,0] - B[0,0])/(W[0,1] - W[1,1])
    w12 = (W[1,0] - W[0,0])/(W[0,1] - W[1,1])

    b23 = (B[2,0] - B[1,0])/(W[1,1] - W[2,1])
    w23 = (W[2,0] - W[1,0])/(W[1,1] - W[2,1])

    x = np.linspace(0,rangeX,10)
    y = w12 * x + b12
    plt.plot(x,y)

    x = np.linspace(0,rangeX,10)
    y = w23 * x + b23
    plt.plot(x,y)

    title = str.format("eta:{0}, iteration:{1}, eps:{2}", eta, iteration, eps)
    plt.title(title)
    
    plt.xlabel("Temperature")
    plt.ylabel("Humidity")
    plt.show()

def Inference(W,B,xt,X_range,X_min):
    xt_normalized = NormalizeByRange(xt, X_range, X_min)
    Z = ForwardCalculation(W,xt_normalized,B)
    A = Softmax(Z)
    r = np.argmax(A)
    return r

XData, Y = LoadData()
X, X_range, X_min = NormalizeByData(XData)
num_features = X.shape[0]
num_samples = X.shape[1]
num_category = Y.shape[0]
prev_loss, loss, diff_loss = 0,0,0
eps=1e-11
W, B = InitialWeights(num_category, num_features, 'zero')
eta = 0.1
max_iteration = 1000

for iteration in range(max_iteration):
    for i in range(num_samples):
        Xm = X[:,i].reshape(num_features,1)
        Ym = Y[:,i].reshape(num_category,1)
        Z = ForwardCalculation(W,Xm,B)
        A = Softmax(Z)
        dw,db = BackPropagation(Xm, Ym, A)
        W,B = UpdateWeights(W, B, dw, db, eta)
        loss, diff_loss = CheckLoss(W,B,X,Y,num_samples,prev_loss)
        if i%10==0:
            print(iteration,i,loss,diff_loss)
        
        if diff_loss < eps:
            print(loss, diff_loss)
            break
        prev_loss = loss

    if diff_loss < eps:
        break
    
ShowResult(X,Y,W,B,np.max(X[0,:]),eta,iteration,eps)
print(W)
print(B)

xt = np.array([5.1,12.7]).reshape(2,1)
result = Inference(W,B,xt,X_range,X_min)
print(result)

xt = np.array([25.3,42.1]).reshape(2,1)
result = Inference(W,B,xt,X_range,X_min)
print(result)

xt = np.array([34.3,82.2]).reshape(2,1)
result = Inference(W,B,xt,X_range,X_min)
print(result)
