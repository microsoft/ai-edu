# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

def target_function2(a,y):
    p1 = y * np.log(a)
    p2 = (1-y) * np.log(1-a)
    y = -p1 - p2
    return y

if __name__ == '__main__':
    err = 1e-2  # avoid invalid math caculation
    a = np.linspace(0+err,1-err)
    y = 0
    z1 = target_function2(a,y)
    y = 1
    z2 = target_function2(a,y)
    p1, = plt.plot(a,z1)
    p2, = plt.plot(a,z2)
    plt.grid()
    plt.legend([p1,p2],["y=0","y=1"])
    plt.xlabel("a")
    plt.ylabel("Loss")
    plt.show()