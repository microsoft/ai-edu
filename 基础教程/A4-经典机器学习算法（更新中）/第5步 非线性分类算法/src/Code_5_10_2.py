
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *




if __name__=="__main__":

    x,y=make_circles(n_samples=100,factor=0.5,noise=0.1)
    print(x,y)

    p = y == 1
    n = y == 0
    fig = plt.figure()

    plt.scatter(x[p,0], x[p,1], marker='^')
    plt.scatter(x[n,0], x[n,1], marker='.')
    plt.show()
