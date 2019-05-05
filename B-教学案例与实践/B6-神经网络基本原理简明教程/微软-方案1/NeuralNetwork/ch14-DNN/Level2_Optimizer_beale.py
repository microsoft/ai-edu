# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from collections import OrderedDict
from MiniFramework.Optimizer import *

def f(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2


def df(x, y):
    dx = 2. * ( (1.5 - x + x * y) * (y - 1) + \
                (2.25 - x + x * y**2) * (y**2 - 1) + \
                (2.625 - x + x * y**3) * (y**3 - 1) )
    dy = 2. * ( (1.5 - x + x * y) * x + \
              (2.25 - x + x * y**2) * 2. * x * y + \
              (2.625 - x + x * y**3) * 3. * x * y**2 )
    return dx,dy


def draw(x_history, y_history):
    x = np.linspace(-4.5, 4.5, 50)
    y = np.linspace(-4.5, 4.5, 50)
    
    X, Y = np.meshgrid(x, y) 
    Z = f(X, Y)

    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection='3d', elev=80, azim=-100)

    #ax.plot_surface(X, Y, Z, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.8, cmap=plt.cm.jet)
    #ax.plot(*minima_, f(*minima_), 'r*', markersize=20)

    #ax.contour(X, Y, Z, levels=np.logspace(-.5, 5, 35))
    ax.contour(X, Y, Z, levels=np.logspace(-.5, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)


    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim((-4.5, 4.5))
    ax.set_ylim((-4.5, 4.5))

    plt.plot(x_history, y_history, 'o-', color="red")


    plt.show()



def run():
    x = 1
    y = 2

    x_history = []
    y_history = []
    
    dict = {OptimizerName.SGD:0.9, OptimizerName.Momentum:0.1, OptimizerName.AdaGrad:1.5, OptimizerName.Adam:0.3}
#    optimizer_x = OptimizerFactory().CreateOptimizer(dict[key], key)
#    optimizer_y = OptimizerFactory().CreateOptimizer(dict[key], key)
    optimizer_x = OptimizerFactory().CreateOptimizer(0.01, OptimizerName.Momentum)
    optimizer_y = OptimizerFactory().CreateOptimizer(0.01, OptimizerName.Momentum)


    for i in range(20):
        x_history.append(x)
        y_history.append(y)
        
        gradx, grady = df(x, y)
        x = optimizer_x.update(x, gradx)
        y = optimizer_y.update(y, grady)
    

    return x_history, y_history

if __name__ == '__main__':
    x_history, y_history = run()
    print(x_history)
    print(y_history)
    draw(x_history, y_history)

