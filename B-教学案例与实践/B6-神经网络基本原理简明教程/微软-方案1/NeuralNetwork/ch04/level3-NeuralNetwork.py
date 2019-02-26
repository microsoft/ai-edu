# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# define funtions to replace flat code, using SGD

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
        # 注意这里和前面的例子不同
        return X.reshape(1,-1),Y.reshape(1,-1)
    else:
        return None,None

def ForwardCalculation(w,b,x):
    z = np.dot(w, x) + b
    return z

def BackPropagation(x,y,z):
    dZ = z - y
    dB = dZ
    dW = np.dot(dZ, x)
    return dW, dB

def UpdateWeights(w, b, dW, dB, eta):
    w = w - eta*dW
    b = b - eta*dB
    return w,b

def Inference(w,b,x):
    z = ForwardCalculation(w,b,x)
    return z

def GetSample(X,Y,i):
    x = X[0,i]
    y = Y[0,i]
    return x,y

def ShowResult(X, Y, w, b, epoch, iteration):
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
    print("epoch=%d,iteration=%d,w=%f,b=%f" %(epoch,iteration,w,b))
    print("w=%f,b=%f" %(w,b))

if __name__ == '__main__':
    # learning rate
    eta = 0.1
    # set w,b=0, you can set to others values to have a try
    #w, b = np.random.random(),np.random.random()
    w, b = 0, 0
    # create mock up data
    X, Y = ReadData()
    # count of samples
    num_example = X.shape[1]

    for i in range(num_example):
        # get x and y value for one sample
        x,y = GetSample(X,Y,i)
        # get z from x,y
        z = ForwardCalculation(w, b, x)
        # calculate gradient of w and b
        dW, dB = BackPropagation(x, y, z)
        # update w,b
        w, b = UpdateWeights(w, b, dW, dB, eta)

    ShowResult(X, Y, w, b, 1, num_example)

    result = Inference(w,b,0.346)
    print("result=", result)