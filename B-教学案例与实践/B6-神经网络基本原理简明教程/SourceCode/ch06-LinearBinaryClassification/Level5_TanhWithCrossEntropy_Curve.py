# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
import sys

from HelperClass.ClassifierFunction import *

# modified cross entropy function for tanh
def target_function2(a,y):
    p1 = (1+y) * np.log((1+a)/2)    # change from y to 1+y, from a to 1+a
    p2 = (1-y) * np.log((1-a)/2)
    y = -p1 - p2
    return y

# modified cross entropy for tanh
def draw_cross_entropy():
    eps = 1e-2
    a = np.linspace(-1+eps,1-eps)
    y = -1  # change from 0 to -1
    z1 = target_function2(a,y)
    y = 1
    z2 = target_function2(a,y)
    p1, = plt.plot(a,z1)
    p2, = plt.plot(a,z2)
    plt.grid()
    plt.legend([p1,p2],["y=-1","y=1"])
    plt.xlabel("a")
    plt.ylabel("Loss")
    plt.show()

def draw_tanh_seperator(fun, label, x, y):
    z = np.linspace(-5,5)
    a = fun.forward(z)
    plt.plot(z,a)
    plt.plot(x,y)
    plt.grid()
    plt.xlabel("input : z")
    plt.ylabel("output : a")
    plt.title(label)
    plt.show()

if __name__ == '__main__':
    draw_cross_entropy()
    draw_tanh_seperator(Tanh(), "Tanh Function", [-5,5], [0,0])
    draw_tanh_seperator(Logistic(), "Logistic Function", [-5,5],[0.5,0.5])
