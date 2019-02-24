# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def target_function(x,y):
    J = x**2 + np.sin(y)**2
    return J

def derivative_function(theta):
    x = theta[0]
    y = theta[1]
    return np.array([2*x,2*np.sin(y)*np.cos(y)])

def show_3d_surface(x, y, z):
    fig = plt.figure()
    ax = Axes3D(fig)
 
    u = np.linspace(-3, 3, 100)
    v = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(u, v)
    R = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            R[i, j] = X[i, j]**2 + np.sin(Y[i, j])**2

    ax.plot_surface(X, Y, R, cmap='rainbow')
    plt.plot(x,y,z,c='black')
    plt.show()

if __name__ == '__main__':
    theta = np.array([3,1])
    eta = 0.1
    error = 1e-2

    X = []
    Y = []
    Z = []
    for i in range(100):
        print(theta)
        x=theta[0]
        y=theta[1]
        z=target_function(x,y)
        X.append(x)
        Y.append(y)
        Z.append(z)
        print("%d: x=%f, y=%f, z=%f" %(i,x,y,z))
        d_theta = derivative_function(theta)
        print("    ",d_theta)
        theta = theta - eta * d_theta
        if z < error:
            break
    show_3d_surface(X,Y,Z)
