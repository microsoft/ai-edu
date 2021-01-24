# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

train_data_name = "../../Data/ch11.train.npz"
test_data_name = "../../Data/ch11.test.npz"

def circle(x1,x2):
    r = np.sqrt(x1*x1 + x2*x2)
    if r < 0.45:
        return True
    else:
        return False

def square(x1,x2):
    if np.abs(x1) < 0.2 and np.abs(x2) < 0.2:
        return True
    return False

def show(X, Y, count):
    fig = plt.figure(figsize=(6,6))
    for i in range(count):
        x1 = X[i,0]
        x2 = X[i,1]
    
        t1 = square(x1,x2)
        t2 = circle(x1,x2)
    
        if t1 == True:
            plt.plot(x1, x2,'s',c='g')
            Y[i] = 1
        elif t1 == False and t2 == True:
            plt.plot(x1, x2,'x',c='r')
            Y[i] = 2
        else:
            plt.plot(x1, x2,'o',c='b')
            Y[i] = 3
        #end if
    #end for
    plt.show()

if __name__ == '__main__':
    c = 1500
    X = np.random.rand(c,2) - 0.5
    Y = np.zeros((c,1))
    show(X[0:1000,:],Y[0:1000,:],1000)
    show(X[1000:,:],Y[1000:,:],500)

    np.savez(train_data_name, data=X[0:1000,:], label=Y[0:1000,:])
    np.savez(test_data_name, data=X[1000:,:], label=Y[1000:,:])
