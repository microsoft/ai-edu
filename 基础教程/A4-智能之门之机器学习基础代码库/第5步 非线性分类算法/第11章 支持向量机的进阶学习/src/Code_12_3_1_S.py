from numpy.core.defchararray import mod
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

from Code_11_2_1_C import *

def main():
    X = np.array([[3,0],[2,4],[3,4],[3,3],[3,2],[3,1],[1,1],[2,2],[2,1],[2,0],[4,1]])
    Y = np.array([1,1,1,1,1,1,-1,-1,-1,-1,-1])

    fig = plt.figure()
    plt.grid()
    plt.axis('equal')
    show_samples(plt, X, Y)
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(131)
    C = 10
    svc(ax, C, X, Y)
    ax = fig.add_subplot(132)
    C = 1
    svc(ax, C, X, Y)
    ax = fig.add_subplot(133)
    C = 0.1
    svc(ax, C, X, Y)
    plt.show()

if __name__=="__main__":
    main()
