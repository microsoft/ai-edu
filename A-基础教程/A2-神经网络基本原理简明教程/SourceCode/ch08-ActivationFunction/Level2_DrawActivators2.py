# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

from Activators.Relu import *
from Activators.Elu import *
from Activators.LeakyRelu import *
from Activators.Sigmoid import *
from Activators.Softplus import *
from Activators.Step import *
from Activators.Tanh import *
from Activators.BenIdentity import *

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

if __name__ == '__main__':
    Draw(-5,5,CRelu(),"Relu Function","Derivative of Relu")
    Draw(-4,4,CElu(0.8),"ELU Function","Derivative of ELU")
    Draw(-5,5,CLeakyRelu(0.01),"Leaky Relu Function","Derivative of Leaky Relu")
    Draw(-5,5,CSoftplus(),"Softplus Function","Derivative of Softplus")
    Draw(-7,7,CBenIdentity(),"BenIdentity Function","Derivative of BenIdentity")
