# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PollutionCategoryCreateData3 import *

x_data_name = "Pollution2CategoryX.dat.npy"
y_data_name = "Pollution2CategoryY.dat.npy"


# remove columns from X/Y where Y == 1
def RemoveCategory3(XRawData, YRawData):
    assert(XRawData.shape[1] == YRawData.shape[1])
    num_example = XRawData.shape[1]
    num_feature = XRawData.shape[0]
    num_category13 = len(np.where(YRawData!=3)[0])
    XData = np.zeros((num_feature, num_category13))
    YData = np.zeros((1,num_category13))
    j = 0
    for i in range(num_example):
        if YRawData[0,i] != 3:
            XData[:,j] = XRawData[:,i]
            YData[:,j] = YRawData[:,i]
            j = j + 1
        # end if
    # end for
    return XData,YData

def Show(X,Y):
    for i in range(X.shape[1]):
        if Y[0,i] == 1:
            plt.scatter(X[0,i], X[1,i], c='r')
        elif Y[0,i] == 2:
            plt.scatter(X[0,i], X[1,i], c='b')
        else:
            plt.scatter(X[0,i], X[1,i], c='g')

    plt.xlabel("Temperature")
    plt.ylabel("Humidity")
    plt.show()


if __name__ == '__main__':
    range_temperature, range_humidity = 40, 80
    X3, Y3 = CreateSampleData(range_temperature, range_humidity, 200)
    X2,Y2=RemoveCategory3(X3,Y3)
    print(X2.shape,Y2.shape)
    np.save("Pollution2CategoryX.dat", X2)
    np.save("Pollution2CategoryY.dat", Y2)
    Show(X2,Y2)
