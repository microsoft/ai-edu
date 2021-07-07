# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
import math

def DrawTwoCategoryPoints(X1, X2, Y, xlabel="x1", ylabel="x2", title=None, show=False, isPredicate=False):
    colors = ['b', 'r']
    shapes = ['s', 'x']
    assert(X1.shape[0] == X2.shape[0] == Y.shape[0])
    count = X1.shape[0]
    for i in range(count):
        j = (int)(round(Y[i]))
        if j < 0:
            j = 0
        if isPredicate:
            plt.scatter(X1[i], X2[i], color=colors[j], marker='^', s=200, zorder=10)
        else:
            plt.scatter(X1[i], X2[i], color=colors[j], marker=shapes[j], zorder=10)
    # end for
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


def DrawThreeCategoryPoints(X1, X2, Y_onehot, xlabel="x1", ylabel="x2", title=None, show=False, isPredicate=False):
    colors = ['b', 'r', 'g']
    shapes = ['s', 'x', 'o']
    assert(X1.shape[0] == X2.shape[0] == Y_onehot.shape[0])
    count = X1.shape[0]
    for i in range(count):
        j = np.argmax(Y_onehot[i])
        if isPredicate:
            plt.scatter(X1[i], X2[i], color=colors[j], marker='^', s=200, zorder=10)
        else:
            plt.scatter(X1[i], X2[i], color=colors[j], marker=shapes[j], zorder=10)
    #end for
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()
