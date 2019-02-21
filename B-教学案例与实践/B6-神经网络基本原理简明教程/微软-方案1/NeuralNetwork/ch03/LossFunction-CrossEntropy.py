# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

def target_function(x):
    p1 = x * np.log(x)
    p2 = (1-x) * np.log(1-x)
    y = -p1 - p2
    return y

if __name__ == '__main__':
    err = 1e-5  # avoid invalid math caculation
    x = np.linspace(0+err,1-err)
    y = target_function(x)
    plt.plot(x,y)
    plt.grid()
    plt.show()