# multiple iteration, loss calculation, stop condition, predication

# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

x_data_name = "TemperatureControlXData.dat"
y_data_name = "TemperatureControlYData.dat"

def ReadData():
    Xfile = Path(x_data_name)
    Yfile = Path(y_data_name)
    if Xfile.exists() & Yfile.exists():
        X = np.load(Xfile)
        Y = np.load(Yfile)
        return X,Y
    else:
        return None,None

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

def ShowResult(X, Y, w, b, iteration):
    # draw sample data
    plt.plot(X, Y, "b.")
    # draw predication data
    Z = w*X +b
    plt.plot(X, Z, "r")
    plt.title("Air Conditioner Power")
    plt.xlabel("Number of Servers(K)")
    plt.ylabel("Power of Air Conditioner(KW)")
    plt.show()
    print(iteration)
    print(w,b)

def Predicate(x, w, b):
    return ForwardCalculation(w, b, x)

if __name__ == '__main__':

    # initialize_data
    eta = 0.01
    # set w,b=0, you can set to others values to have a try
    #w, b = np.random.random(),np.random.random()
    w, b = 0, 0
    eps = 1e-10
    iteration, max_iteration = 0, 100
    # calculate loss to decide the stop condition
    prev_loss, loss, diff_loss = 0,0,0
    # create mock up data
    X, Y = ReadData()
    # count of samples
    num_example = X.shape[0]

    for iteration in range(max_iteration):
        for i in range(num_example):
            # get x and y value for one sample
            x = X[i]
            y = Y[i]
            # get z from x,y
            z = ForwardCalculation(w, b, x)
            # calculate gradient of w and b
            dW, dB = BackPropagation(x, y, z)
            # update w,b
            w, b = UpdateWeights(w, b, dW, dB, eta)
            # calculate loss for this batch
            loss, diff_loss = CheckLoss(w,b,X,Y,num_example,prev_loss)
            if i%10==0:
                print(iteration,i,loss,diff_loss,w,b)
            # condition 1 to stop
            if diff_loss < eps:
                break
            prev_loss = loss

        if diff_loss < eps:
            break

    print(loss, diff_loss)
    ShowResult(X, Y, w, b, iteration)

    x = 346/1000
    result = Predicate(x, w, b)
    print(result)
