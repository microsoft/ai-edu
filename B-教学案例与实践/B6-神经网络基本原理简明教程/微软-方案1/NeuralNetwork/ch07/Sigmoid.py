# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

class CSigmoid(object):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a

    def backward(self, z, a, delta):
        da = np.multiply(a, 1-a)
        dz = np.multiply(delta, da)
        return da, dz

def Draw(start,end,func):
    z = np.linspace(start, end, 200)
    a = func.forward(z)
    plt.plot(z,a)
    plt.grid()
    plt.xlabel("input : z")
    plt.ylabel("output : a")
    plt.title("Sigmoid")

    plt.plot([-7,7],[0.5,0.5],'-')

    plt.show()


if __name__ == '__main__':
    Draw(-7,7,CSigmoid())
