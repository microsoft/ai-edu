# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def LoadData():
    Xfile = Path("TreeWormXData.dat")
    Yfile = Path("TreeWormYData.dat")
    if Xfile.exists() & Yfile.exists():
        XData = np.load(Xfile)
        YData = np.load(Yfile)
        return XData,YData
    
    return None,None

def NormalizeData(X):
    X_NEW = np.zeros(X.shape)
    # get number of features
    n = X.shape[0]
    for i in range(n):
        x_row = X[i,:]
        x_max = np.max(x_row)
        x_min = np.min(x_row)
        if x_max != x_min:
            x_new = (x_row - x_min)/(x_max-x_min)
            X_NEW[i,:] = x_new
    return X_NEW

def ForwardCalculation(X, dict_Param):
    W1 = dict_Param["W1"]
    B1 = dict_Param["B1"]
    W2 = dict_Param["W2"]
    B2 = dict_Param["B2"]
    
    Z1 = np.dot(W1,X)+B1
    A1 = Sigmoid(Z1)
    #A1 = np.tanh(Z1)

    Z2=np.dot(W2,A1)+B2
    A2=Sigmoid(Z2)
    
    dict_Cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return A2, dict_Cache

def BackPropagation(dict_Param,cache,X,Y):
    W1=dict_Param["W1"]
    W2=dict_Param["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1=cache["Z1"]

    dZ2= A2 - Y
    dW2 = np.dot(dZ2, A1.T)
    dB2 = np.sum(dZ2, axis=1, keepdims=True)

    dLoss_A1 = np.dot(W2.T, dZ2)
    dA1_Z1 = A1 * (1 - A1)     # sigmoid
    #dA1_Z1 = 1-np.power(A1,2)   # tanh
    dZ1 = dLoss_A1 * dA1_Z1
    
    dW1 = np.dot(dZ1, X.T)
    dB1 = np.sum(dZ1, axis=1, keepdims=True)

    dict_Grads = {"dW1": dW1, "dB1": dB1, "dW2": dW2, "dB2": dB2}
    return dict_Grads

def UpdateParam(dict_Param, dict_Grads, learning_rate):
    W1 = dict_Param["W1"]
    B1 = dict_Param["B1"]
    W2 = dict_Param["W2"]
    B2 = dict_Param["B2"]

    dW1 = dict_Grads["dW1"]
    dB1 = dict_Grads["dB1"]
    dW2 = dict_Grads["dW2"]
    dB2 = dict_Grads["dB2"]

    W1 = W1 - learning_rate * dW1
    B1 = B1 - learning_rate * dB1
    W2 = W2 - learning_rate * dW2
    B2 = B2 - learning_rate * dB2

    dict_Param = {"W1": W1, "B1": B1, "W2": W2, "B2": B2}
    return dict_Param

# cross entropy: -Y*lnA
def CalculateLoss(dict_Param, X, Y, count):
    A2, dict_Cache = ForwardCalculation(X, dict_Param)
    p = Y * np.log(A2) + (1-Y) * np.log(1-A2)
    Loss = -np.sum(p) / count
    return Loss

def InitialParameters(num_input, num_hidden, num_output, flag):
    if flag == 0:
        # zero
        W1 = np.zeros((num_hidden, num_input))
        W2 = np.zeros((num_output, num_hidden))
    elif flag == 1:
        # normalize
        W1 = np.random.normal(size=(num_hidden, num_input))
        W2 = np.random.normal(size=(num_output, num_hidden))
    elif flag == 2:
        #
        W1=np.random.uniform(-np.sqrt(6)/np.sqrt(num_input+num_hidden),np.sqrt(6)/np.sqrt(num_hidden+num_input),size=(num_hidden,num_input))
        W2=np.random.uniform(-np.sqrt(6)/np.sqrt(num_output+num_hidden),np.sqrt(6)/np.sqrt(num_output+num_hidden),size=(num_output,num_hidden))

    B1 = np.zeros((num_hidden, 1))
    B2 = np.zeros((num_output, 1))
    dict_Param = {"W1": W1, "B1": B1, "W2": W2, "B2": B2}
    return dict_Param

def Sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

def ShowResult(X, Y):
    XT = FindBoundary(dict_Param)
    plt.plot(XT[0,:],XT[1,:],'g')

    for i in range(X.shape[1]):
        x1 = X[0,i]
        x2 = X[1,i]
        if Y[i] == 0:
            plt.scatter(x1, x2, c='r')
        else:
            plt.scatter(x1, x2, c='b')

    plt.xlabel("Years")
    plt.ylabel("Seasons")
    plt.title("Worm Probability")
    plt.show()

def FindBoundary(dict_Param):
    count = 100
    XT = np.zeros((2,count))
    XT[0,:] = np.linspace(0,1,num=count)
    for i in range(count):
        prev_diff = 10
        Y = np.linspace(0,1,num=count)
        for j in range(count):
            XT[1,i] = Y[j]
            A2, cache = ForwardCalculation(XT[:,i].reshape(2,1), dict_Param)
            current_diff = np.abs(A2[0][0] - 0.5)
            if current_diff > prev_diff:
                break
            prev_diff = current_diff

    return XT

print("Loading...")
learning_rate = 0.1
num_hidden = 10
num_output = 1
raw_data,Y = LoadData()
X = NormalizeData(raw_data)

num_images = X.shape[1]
num_input = X.shape[0]
max_iteration = 1000

dict_Param = InitialParameters(num_input, num_hidden, num_output, 2)
prev_loss = 0
diff_loss = 10
eps = 1e-10
print("Training...")
for iteration in range(max_iteration):
    for item in range(num_images):
        x = X[:,item].reshape(num_input,1)
        y = Y[item]
        A2, dict_Cache = ForwardCalculation(x, dict_Param)
        dict_Grads = BackPropagation(dict_Param, dict_Cache, x, y)
        dict_Param = UpdateParam(dict_Param, dict_Grads, learning_rate)
        Loss = CalculateLoss(dict_Param, X, Y, num_images)
        diff_loss = np.abs(Loss-prev_loss)
        prev_loss = Loss
        if diff_loss < eps:
            break
    print(iteration, Loss)
    if diff_loss < eps:
        break
 
print("complete")
print("W1:", dict_Param["W1"])
print("B1:", dict_Param["B1"])
print("W2:", dict_Param["W2"])
print("B2:", dict_Param["B2"])

print("testing...")

ShowResult(X, Y)
