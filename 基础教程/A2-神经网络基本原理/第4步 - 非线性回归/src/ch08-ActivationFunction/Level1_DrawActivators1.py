# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from Activators.Relu import *
from Activators.Elu import *
from Activators.LeakyRelu import *
from Activators.Sigmoid import *
from Activators.Softplus import *
from Activators.Step import *
from Activators.Tanh import *
from Activators.BenIdentity import *

mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus']=False

def Draw(start,end,func,lable1,lable2):
    z = np.linspace(start, end, 200)
    a = func.forward(z)
    da, dz = func.backward(z, a, 1)

    p1, = plt.plot(z,a)
    p2, = plt.plot(z,da)
    plt.legend([p1,p2], [lable1, lable2])
    plt.grid()
    plt.xlabel("输入 : $z$")
    plt.ylabel("输出 : $a$")
    plt.title(lable1)
    plt.show()

if __name__ == '__main__':
    Draw(-7,7,CSigmoid(),"Sigmoid 函数","导数")
    Draw(-7,7,CSigmoid(),"Logistic 函数","导数")
    Draw(-7,7,CTanh(),"Tanh 函数","导数")
    Draw(-5,5,CStep(0.3),"Step 函数","导数")
