# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

from HelperClass.NeuralNet_1_2 import *
from HelperClass.Visualizer_1_0 import *

file_name = "../../data/ch07.npz"

def ShowData(X,Y):
    fig = plt.figure(figsize=(6,6))
    DrawThreeCategoryPoints(X[:,0], X[:,1], Y[:], xlabel="x1", ylabel="x2", show=True)

def ShowResult(X,Y,xt,yt):
    fig = plt.figure(figsize=(6,6))
    DrawThreeCategoryPoints(X[:,0], X[:,1], Y[:], xlabel="x1", ylabel="x2", show=False)

    b13 = (net.B[0,0] - net.B[0,2])/(net.W[1,2] - net.W[1,0])
    w13 = (net.W[0,0] - net.W[0,2])/(net.W[1,2] - net.W[1,0])

    b23 = (net.B[0,2] - net.B[0,1])/(net.W[1,1] - net.W[1,2])
    w23 = (net.W[0,2] - net.W[0,1])/(net.W[1,1] - net.W[1,2])

    b12 = (net.B[0,1] - net.B[0,0])/(net.W[1,0] - net.W[1,1])
    w12 = (net.W[0,1] - net.W[0,0])/(net.W[1,0] - net.W[1,1])

    x = np.linspace(0,1,2)
    y = w13 * x + b13
    p13, = plt.plot(x,y,c='r')

    x = np.linspace(0,1,2)
    y = w23 * x + b23
    p23, = plt.plot(x,y,c='b')

    x = np.linspace(0,1,2)
    y = w12 * x + b12
    p12, = plt.plot(x,y,c='g')

    plt.legend([p13,p23,p12], ["13","23","12"])
    plt.axis([-0.1,1.1,-0.1,1.1])

    DrawThreeCategoryPoints(xt[:,0], xt[:,1], yt[:], xlabel="x1", ylabel="x2", show=True, isPredicate=True)

# 主程序
if __name__ == '__main__':
    num_category = 3
    reader = DataReader_1_3(file_name)
    reader.ReadData()
    reader.ToOneHot(num_category, base=1)
    # show raw data before normalization
    ShowData(reader.XRaw, reader.YTrain)
    reader.NormalizeX()

    num_input = 2
    params = HyperParameters_1_1(num_input, num_category, eta=0.1, max_epoch=100, batch_size=10, eps=1e-3, net_type=NetType.MultipleClassifier)
    net = NeuralNet_1_2(params)
    net.train(reader, checkpoint=1)

    xt_raw = np.array([5,1,7,6,5,6,2,7]).reshape(4,2)
    xt = reader.NormalizePredicateData(xt_raw)
    output = net.inference(xt)
    print(output)

    ShowResult(reader.XTrain, reader.YTrain, xt, output)
