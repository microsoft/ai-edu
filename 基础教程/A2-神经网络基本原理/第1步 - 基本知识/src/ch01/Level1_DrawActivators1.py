# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus']=False

class CSigmoid(object):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a


def Draw(start,end,func,lable1):
    z = np.linspace(start, end, 200)
    a = func.forward(z)

    p1, = plt.plot(z,a)
    plt.grid()
    plt.xlabel("$Z$（input）")
    plt.ylabel("$A$（output）")
    plt.title(lable1)
    plt.show()

if __name__ == '__main__':
    Draw(-7,7,CSigmoid(),"激活函数图像")
