# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from MiniFramework.Optimizer import *


def f(x, y):
    return x**2 / 20.0 + y**2


def df(x, y):
    return x / 10.0, 2.0*y



init_pos = (-7.0, 2.0)
params = {}
x, y = init_pos[0], init_pos[1]
gradx, grady = 0, 0



dict = {OptimizerName.SGD:0.9, OptimizerName.Momentum:0.1, OptimizerName.AdaGrad:1.5, OptimizerName.Adam:0.3}
idx = 1

for key in dict.keys():
    optimizer_x = OptimizerFactory().CreateOptimizer(dict[key], key)
    optimizer_y = OptimizerFactory().CreateOptimizer(dict[key], key)
    x_history = []
    y_history = []
    x,y = init_pos[0], init_pos[1]
    
    for i in range(20):
        x_history.append(x)
        y_history.append(y)
        
        gradx, grady = df(x, y)
        x = optimizer_x.update(x, gradx)
        y = optimizer_y.update(y, grady)
    

    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    
    X, Y = np.meshgrid(x, y) 
    Z = f(X, Y)
    
    # for simple contour line  
    mask = Z > 7
    Z[mask] = 0
    
    # plot 
    plt.subplot(2, 2, idx)
    idx += 1
    plt.plot(x_history, y_history, 'o-', color="red")
    plt.contour(X, Y, Z)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    #colorbar()
    #spring()
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")
    
plt.show()