# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np

# x=0,y=1; x=1,y=0
def ReadData():
    X = np.array([0,1]).reshape(1,2)
    Y = np.array([1,0]).reshape(1,2)
    return X,Y

def ForwardCalculation(w,b,x):
    z = w * x + b
    return z

def BackPropagation(x,y,z):
    dZ = z - y
    dB = dZ
    dW = dZ * x
    return dW, dB

def UpdateWeights(w, b, dW, dB, eta):
    w = w - eta*dW
    b = b - eta*dB
    return w,b

def CheckLoss(w, b, X, Y, count, prev_loss):
    Z = w * X + b
    LOSS = (Z - Y)**2
    loss = LOSS.sum()/count/2
    diff_loss = abs(loss - prev_loss)
    return loss, diff_loss

def Test(w,b):
    z1 = ForwardCalculation(w, b, 0)
    z2 = ForwardCalculation(w, b, 1)
    print (z1,z2)
    if np.abs(z1-1) < 0.001 and np.abs(z2-0)<0.001:
        return True
    return False

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


# initialize_data
eta = 0.1
# set w,b=0, you can set to others values to have a try
#w, b = np.random.random(),np.random.random()
w, b = 0, 0
eps = 1e-10
iteration, max_iteration = 0, 1000
# calculate loss to decide the stop condition
prev_loss, loss, diff_loss = 0,0,0
# create mock up data
X, Y = ReadData()
# count of samples
num_features = X.shape[0]
num_example = X.shape[1]

for iteration in range(max_iteration):
    for i in range(num_example):
        # get x and y value for one sample
        x = X[:,i]
        y = Y[:,i]
        # get z from x,y
        z = ForwardCalculation(w, b, x)
        # calculate gradient of w and b
        dW, dB = BackPropagation(x, y, z)
        # update w,b
        w, b = UpdateWeights(w, b, dW, dB, eta)
        # calculate loss for this batch
        loss, diff_loss = CheckLoss(w,b,X,Y,num_example,prev_loss)
        print(iteration,i,loss,diff_loss,w,b)
        # condition 1 to stop
        if diff_loss < eps:
            break
        prev_loss = loss

    if diff_loss < eps:
        break

print("w=%f,b=%f" %(w,b))

# test
print(Test(w,b))

