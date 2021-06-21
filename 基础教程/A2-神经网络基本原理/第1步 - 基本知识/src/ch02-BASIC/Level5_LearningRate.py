# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

def targetFunction(x):
    y = (x-1)**2 + 0.1
    return y

def derivativeFun(x):
    y = 2*(x-1)
    return y

def create_sample():
    x = np.linspace(-1,3,num=100)
    y = targetFunction(x)
    return x,y

def draw_base():
    x,y=create_sample()
    plt.plot(x,y,'.')
    plt.show()
    return x,y
   
def gd(eta):
    x = -0.8
    a = np.zeros((2,10))
    for i in range(10):
        a[0,i] = x
        a[1,i] = targetFunction(x)
        dx = derivativeFun(x)
        x = x - eta*dx
    
    plt.plot(a[0,:],a[1,:],'x')
    plt.plot(a[0,:],a[1,:])
    plt.title("eta=%f" %eta)
    plt.show()

if __name__ == '__main__':

    eta = [1.1,1.,0.8,0.6,0.4,0.2,0.1]

    for e in eta:
        X,Y=create_sample()
        plt.plot(X,Y,'.')
        #plt.show()
        gd(e)
