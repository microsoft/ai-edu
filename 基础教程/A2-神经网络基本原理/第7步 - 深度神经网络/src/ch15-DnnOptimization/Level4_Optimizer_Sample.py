# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

from MiniFramework.Optimizer_1_0 import *

def f(x, y):
    return x**2 / 10.0 + y**2

def derivative_f(x, y):
    return x / 5.0, 2.0*y


def run():
    dict = {OptimizerName.SGD:0.95, OptimizerName.Momentum:0.05, OptimizerName.RMSProp:0.5, OptimizerName.Adam:0.3}
    idx = 1
    fig = plt.figure(figsize=(12,9))

    for key in dict.keys():
        optimizer_x = OptimizerFactory().CreateOptimizer(dict[key], key)
        optimizer_y = OptimizerFactory().CreateOptimizer(dict[key], key)
        x_history = []
        y_history = []
        x,y = -7.0, 2.0
        dx, dy = 0, 0

        for i in range(25):
            x_history.append(x)
            y_history.append(y)
        
            dx, dy = derivative_f(x, y)
            x = optimizer_x.update(x, dx)
            y = optimizer_y.update(y, dy)
        # end for    

        x = np.arange(-10, 10, 0.01)
        y = np.arange(-5, 5, 0.01)
    
        X, Y = np.meshgrid(x, y) 
        Z = f(X, Y)
    
        # for simple contour line  
        #mask = Z > 7
        #Z[mask] = 0
    
        # plot 
        plt.subplot(2, 2, idx)
        idx += 1
        plt.plot(x_history, y_history, 'o-')
        c = plt.contour(X, Y, Z)
        plt.ylim(-10, 10)
        plt.xlim(-10, 10)
        plt.plot(0, 0, '+')
        plt.title(key.name)
        plt.xlabel("x")
        plt.ylabel("y")
    # end for

if __name__ == '__main__':
    run()
    plt.suptitle("Optimazers")    
    plt.show()