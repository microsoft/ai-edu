# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

file_name = "../../data/ch07.npz"

def fun1(x):
    y = 2.5 * x -10
    return y

def fun2(x):
    y = 0.3 * x + 5
    return y

def fun3(x):
    y = -1 * x + 10
    return y

if __name__ == '__main__':
    plt.plot([0,10],[-10,15])
    plt.plot([0,10],[5,8])
    plt.plot([0,10],[10,0])

    plt.axis([-0.1,10.1,-0.1,10.1])

    X = np.random.random((200,2)) * 10
    Y = np.zeros((200,1))
    
    for i in range(200):
        x1 = X[i,0]
        x2 = X[i,1]
        y1 = fun1(x1)
        y2 = fun2(x1)
        y3 = fun3(x1)

        noise = (np.random.rand() - 0.5) * 2
        x2 = abs(x2-noise)
        if x2 > y1 and x2 > y2 and x2 > y3:
            Y[i,0] = 1
        elif x2 > y1 and x2 < y2 and x2 < y3:
            Y[i,0] = 2
        elif x2 < y1 and x2 < y2 and x2 > y3:
            Y[i,0] = 3
        else:
            Y[i,0] = 0

    a = np.where(Y > 0)
    count = len(a[0])

    XData = np.zeros((count,2))
    YData = np.zeros((count,1))

    j = 0
    for i in range(200):
        if Y[i,0] == 1:
            plt.plot(X[i,0],X[i,1],'.', c='r')
        elif Y[i,0] == 2:
            plt.plot(X[i,0],X[i,1],'x', c='g')
        elif Y[i,0] == 3:
            plt.plot(X[i,0],X[i,1], 'o', c='b')
        else:
            continue
        XData[j,:] = X[i,:]
        YData[j,:] = Y[i,:]
        j = j + 1
    plt.show()

    for i in range(j):
        if YData[i,0] == 1:
            plt.plot(XData[i,0],XData[i,1],'.', c='r')
        elif YData[i,0] == 2:
            plt.plot(XData[i,0],XData[i,1],'x', c='g')
        elif YData[i,0] == 3:
            plt.plot(XData[i,0],XData[i,1], 'o', c='b')

    plt.show()

    np.savez(file_name, data=XData, label=YData)
