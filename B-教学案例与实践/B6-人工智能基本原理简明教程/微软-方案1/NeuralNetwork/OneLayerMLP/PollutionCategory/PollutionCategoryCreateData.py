# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

x_data_name = "PollutionCategoryXData.dat"
y_data_name = "PollutionCategoryYData.dat"

def TargetFunction(x):
    y1 = 0.52 * x + 50
    y2 = -0.3 * x + 35
    return y1, y2 

def CreateSampleData(range_temperature, range_humidity, count):
    Xfile = Path(x_data_name)
    Yfile = Path(y_data_name)
    if Xfile.exists() & Yfile.exists():
        XData = np.load(Xfile)
        YData = np.load(Yfile)
    else:
        X1 = np.random.rand(count) * range_temperature
        X2 = np.random.rand(count) * range_humidity
        XData = np.zeros((2,count))
        XData[0,:] = X1 
        XData[1,:] = X2
        YData = np.zeros((3,count))
        for i in range(count):
            x = XData[0,i]
            y = XData[1,i]
            y1, y2 = TargetFunction(x)
            if y < y2:
                YData[0,i] = 1
            elif y > y1:
                YData[2,i] = 1
            else:
                YData[1,i] = 1
        np.save(x_data_name, XData)
        np.save(y_data_name, YData)

    return XData, YData

def Show(X,Y):
    for i in range(X.shape[1]):
        if Y[0,i] == 0 and Y[1,i]==0 and Y[2,i]==1:
            plt.scatter(X[0,i], X[1,i], c='r')
        elif Y[0,i] == 0 and Y[1,i]==1 and Y[2,i]==0:
            plt.scatter(X[0,i], X[1,i], c='b')
        else:
            plt.scatter(X[0,i], X[1,i], c='g')

    plt.xlabel("Temperature")
    plt.ylabel("Humidity")
    plt.show()

range_temperature, range_humidity = 40, 80
X, Y = CreateSampleData(range_temperature, range_humidity, 200)
Show(X,Y)
