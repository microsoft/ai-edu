# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

def Draw(start,end,func,lable1,lable2):
    z = np.linspace(start, end, 200)
    a = func.forward(z)
    da, dz = func.backward(z, a, 1)

    p1, = plt.plot(z,a)
    p2, = plt.plot(z,da)
    plt.legend([p1,p2], [lable1, lable2])
    plt.grid()
    plt.xlabel("input : z")
    plt.ylabel("output : a")
    plt.title(lable1)
    plt.show()
