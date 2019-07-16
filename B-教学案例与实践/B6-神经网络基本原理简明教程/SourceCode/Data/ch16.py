# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

train_data_name = "../../data/ch16.train.npz"
test_data_name = "../../data/ch16.test.npz"

def TargetFunction(x):
    p1 = 4*np.sin(x)/3.1416
    #p2 = 4*np.sin(3*x)/3/3.1416
    p2 = 0
    p3 = 4*np.sin(5*x)/5/3.1416
    y1 = p1 + p2 + p3
    y2 = p1
    return y1, y2

def CreateSampleData(num_train, num_test):
    # create train data
    x1 = np.linspace(0,2*3.1416,num_train).reshape(num_train,1)
    y1,y2 = TargetFunction(x1)
    np.savez(train_data_name, data=x1, label=y1)

    # create test data
    x1 = np.linspace(0,2*3.1416,num_test).reshape(num_test,1)
    y1,y2 = TargetFunction(x1)
    np.savez(test_data_name, data=x1, label=y2)

def GetSampleData():
    Trainfile = Path(train_data_name)
    Testfile = Path(test_data_name)
    if Trainfile.exists() & Testfile.exists():
        TrainData = np.load(Trainfile)
        TestData = np.load(Testfile)
        return TrainData, TestData

if __name__ == '__main__':
    CreateSampleData(25, 100)
    TrainData, TestData = GetSampleData()
    plt.scatter(TrainData["data"], TrainData["label"], c='b')
    plt.scatter(TestData["data"], TestData["label"], s=1, c='g')
    plt.show()

