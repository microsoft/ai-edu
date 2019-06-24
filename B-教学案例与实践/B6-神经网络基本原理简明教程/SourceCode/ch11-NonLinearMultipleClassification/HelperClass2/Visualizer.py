# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from HelperClass2.HyperParameters2 import *

def ShowFittingResult(net, count):
    pass



def ShowClassificationResult25D(net, count, title):
    x = np.linspace(0,1,count)
    y = np.linspace(0,1,count)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((count, count))
    input = np.hstack((X.ravel().reshape(count*count,1),Y.ravel().reshape(count*count,1)))
    output = net.inference(input)
    if net.hp.net_type == NetType.BinaryClassifier:
        Z = output.reshape(count,count)
    elif net.hp.net_type == NetType.MultipleClassifier:
        sm = np.argmax(output, axis=1)
        Z = sm.reshape(count,count)
    plt.contourf(X, Y, Z, cmap=plt.cm.Spectral, zorder=1)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)

def ShowDataByOneHot2D(X1, X2, Y, title):
    colors = ['b', 'r', 'g']
    shapes = ['s', 'x', 'o']
    assert(X1.shape[0] == X2.shape[0] == Y.shape[0])
    count = X1.shape[0]
    for i in range(count):
        for j in range(Y.shape[1]):
            if Y[i,j] == 1:
                plt.scatter(X1[i], X2[i], color=colors[j], marker=shapes[j], zorder=10)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    #end for