# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from MiniFramework.ActivationLayer import *

def nd_fun(x, sigma, mu):
    a = - (x-mu)**2 / (2*sigma*sigma)
    f = np.exp(a) / (sigma * np.sqrt(2*np.pi))
    return f

if __name__ == '__main__':
    x = np.linspace(-5,5)
    f = nd_fun(x, 1, 0)
    p1, = plt.plot(x,f)

    f = nd_fun(x, 1.5, 0)
    p2, = plt.plot(x,f)

    f = nd_fun(x, 1.5, 2)
    p3, = plt.plot(x,f)

    plt.legend([p1,p2,p3], ["μ=0,σ=1", "μ=0,σ=1.5", "μ=2,σ=1.5"])

    plt.grid()
    plt.show()

    x = np.linspace(-5,5)
    f = nd_fun(x, 1.5, 2)
    p1, = plt.plot(x,f)
    s = Sigmoid().forward(x)
    p2, = plt.plot(x, s)
    plt.grid()
    plt.legend([p1,p2], ["forward batch data", "Sigmoid"])
    plt.axis([-5, 5, 0, 1])
    plt.show()

    x = np.linspace(-5,5)
    f = nd_fun(x, 1.5, 2)
    p1, = plt.plot(x,f)
    r = Relu().forward(x)
    p2, = plt.plot(x, r)
    plt.grid()
    plt.legend([p1,p2], ["forward batch data", "Relu"])
    plt.axis([-5, 5, 0, 4])
    plt.show()

    x = np.linspace(-5,5)
    s = Sigmoid().forward(x)
    plt.plot(x, s)
    plt.grid()
    plt.show()

    x = np.linspace(-3,3)
    f = nd_fun(x, 1, 0)
    plt.plot(x,f)
    plt.grid()
    plt.show()
