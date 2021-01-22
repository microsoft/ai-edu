# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

train_data_name = "../../Data/ch10.train.npz"
test_data_name = "../../Data/ch10.test.npz"

def create_data(count, title):
    fig = plt.figure(figsize=(6,6))

    x1 = np.linspace(0,3.14,count).reshape(count,1)
    noise = (np.random.random(count)-0.5)/4
    y1 = (np.sin(x1[:,0]) + noise).reshape(count,1)
    plt.scatter(x1,y1)
    X0 = np.hstack((x1,y1))
    Y0 = np.zeros((count,1))

    x2 = np.linspace(0,3.14,count).reshape(count,1)
    noise = (np.random.random(count)-0.5)/4
    y2 = (np.sin(x2[:,0]) + noise + 0.65).reshape(count,1)
    plt.scatter(x2, y2)
    X1 = np.hstack((x2,y2))
    Y1 = np.ones((count,1))

    plt.title(title)
    plt.grid()
    plt.show()

    X = np.concatenate((X0,X1))
    Y = np.concatenate((Y0,Y1))
    return X,Y


def ShowData():
    fig = plt.figure(figsize=(6,6))
    X0 = dr.GetSetByLabel("train", 0)
    X1 = dr.GetSetByLabel("train", 1)
    plt.scatter(X0[:,0], X0[:,1], marker='x', color='r')
    plt.scatter(X1[:,0], X1[:,1], marker='.', color='b')
    plt.show()

if __name__ == '__main__':
    x1,y1=create_data(200, "train data")
    x2,y2=create_data(50, "test data")
    np.savez(train_data_name, data=x1, label=y1)    
    np.savez(test_data_name, data=x2, label=y2)

