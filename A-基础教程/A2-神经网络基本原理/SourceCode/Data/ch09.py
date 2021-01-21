# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

train_data_name = "../../data/ch09.train.npz"
test_data_name = "../../data/ch09.test.npz"

def TargetFunction(x):
    p1 = 0.4 * (x**2)
    p2 = 0.3 * x * np.sin(15 * x)
    p3 = 0.01 * np.cos(50 * x)
    y = p1 + p2 + p3 - 0.3
    return y

def CreateSampleData(num_train, num_test):
    # create train data
    x1 = np.random.random((num_train,1))
    y1 = TargetFunction(x1) + (np.random.random((num_train,1))-0.5)/10
    np.savez(train_data_name, data=x1, label=y1)

    # create test data
    x2 = np.linspace(0,1,num_test).reshape(num_test,1)
    y2 = TargetFunction(x2)
    np.savez(test_data_name, data=x2, label=y2)

def GetSampleData():
    Trainfile = Path(train_data_name)
    Testfile = Path(test_data_name)
    if Trainfile.exists() & Testfile.exists():
        TrainData = np.load(Trainfile)
        TestData = np.load(Testfile)
        return TrainData, TestData

if __name__ == '__main__':
    CreateSampleData(1000, 200)
    TrainData, TestData = GetSampleData()
    plt.scatter(TrainData["data"], TrainData["label"], s=1, c='b')
    plt.scatter(TestData["data"], TestData["label"], s=4, c='r')
    plt.show()

