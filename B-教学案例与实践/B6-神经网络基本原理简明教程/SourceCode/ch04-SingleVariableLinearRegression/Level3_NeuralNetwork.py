# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# define funtions to replace flat code, using SGD, single example for each iteration

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SimpleDataReader import *

file_name = "../../data/ch04.npz"

def ForwardCalculation(w,b,x):
    z = np.dot(x, w) + b
    return z

def BackPropagation(x,y,z):
    dZ = z - y
    dB = dZ
    dW = np.dot(x, dZ)
    return dW, dB

def UpdateWeights(w, b, dW, dB, eta):
    w = w - eta*dW
    b = b - eta*dB
    return w,b

def ShowResult(dataReader, w, b):
    X,Y = dataReader.GetWholeTrainSamples()
    # draw sample data
    plt.plot(X, Y, "b.")
    # draw predication data
    PX = np.linspace(0,1,10)
    PZ = np.dot(PX, w) + b
    plt.plot(PX, PZ, "r")
    plt.title("Air Conditioner Power")
    plt.xlabel("Number of Servers(K)")
    plt.ylabel("Power of Air Conditioner(KW)")
    plt.show()
    print("w=%f,b=%f" %(w,b))

if __name__ == '__main__':

    sdr = SimpleDataReader(file_name)
    sdr.ReadData()

    # learning rate
    eta = 0.1
    # set w,b=0, you can set to others values to have a try
    #w, b = np.random.random(),np.random.random()
    w, b = 0, 0

    for i in range(sdr.num_train):
        # get x and y value for one sample
        x,y = sdr.GetSingleTrainSample(i)
        # get z from x,y
        z = ForwardCalculation(w, b, x)
        # calculate gradient of w and b
        dW, dB = BackPropagation(x, y, z)
        # update w,b
        w, b = UpdateWeights(w, b, dW, dB, eta)

    ShowResult(sdr, w, b)

    # predication
    result = ForwardCalculation(w,b,0.346)
    print("result=", result)