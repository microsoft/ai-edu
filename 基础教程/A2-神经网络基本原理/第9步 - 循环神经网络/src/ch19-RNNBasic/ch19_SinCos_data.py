# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import os
import numpy as np
import matplotlib.pyplot as plt


def get_One_train_data():
    start = (np.random.rand()-0.5)*10
    x = np.linspace(start,start+np.pi,num = 10)
    sinx = np.sin(x)
    cosx = np.cos(x)
    coslastx = cosx[-1]
    return sinx,coslastx


if __name__=='__main__':

    x,y = get_One_train_data()
    print(x)
    print(y)
    exit()

    steps = np.linspace(0, np.pi*2, 100)
    x = np.sin(steps)
    y = np.cos(steps)
    plt.plot(x)
    plt.plot(y)
    plt.show()


