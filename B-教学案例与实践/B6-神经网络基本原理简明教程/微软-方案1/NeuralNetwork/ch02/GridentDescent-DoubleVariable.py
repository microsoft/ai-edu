# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

def target_function(x,y):
    z = x*x + y*y
    return z

def derivative_function(theta):
    x = theta[0]
    y = theta[1]
    return np.array([2*x,2*y])

def draw_function():
    x = np.linspace(-1.2,1.2)
    y = target_function(x)
    plt.plot(x,y)

def draw_gd(X):
    Y = []
    for i in range(len(X)):
        Y.append(target_function(X[i]))
    
    plt.plot(X,Y)

if __name__ == '__main__':
    theta = np.array([1,3])
    eta = 0.1
    error = 1e-2

    for i in range(100):
        theta = theta - eta * derivative_function(theta)
        print(theta)


