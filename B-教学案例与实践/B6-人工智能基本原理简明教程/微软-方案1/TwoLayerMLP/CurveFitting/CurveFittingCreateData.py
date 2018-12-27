# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

train_data_name = "CurveFittingTrainData.dat"
test_data_name = "CurveFittingTestData.dat"

def TargetFunction(x):
    p1 = 0.4 * (x**2)
    p2 = 0.3 * x * np.sin(15 * x)
    p3 = 0.01 * np.cos(50 * x)
    y = p1 + p2 + p3 - 0.3
    return y

def CreateSampleData(n, flag):
    Data = np.zeros((2,n))
    Data[0,:] = np.random.random((1,n))
    if flag==True:  # add noise
        Data[1,:] = TargetFunction(Data[0,:]) + (np.random.random((1,n))-0.5)/10
    else:   # don't add noise
        Data[1,:] = TargetFunction(Data[0,:])
    return Data

def GetSampleData():
    Trainfile = Path(train_data_name)
    Testfile = Path(test_data_name)
    if Trainfile.exists() & Testfile.exists():
        TrainData = np.load(Trainfile)
        TestData = np.load(Testfile)
        return TrainData, TestData
    else:
        TrainData = CreateSampleData(1000, True)
        TestData = CreateSampleData(200, False)
        np.save(train_data_name, TrainData)
        np.save(test_data_name, TestData)

    return TrainData, TestData

TrainData, TestData = GetSampleData()
plt.scatter(TrainData[0,:], TrainData[1,:], s=1)
#plt.scatter(TestData[0,:], TestData[1,:])
plt.show()
