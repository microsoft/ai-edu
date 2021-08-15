import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
import matplotlib as mpl


def fun(X1,X2,a1,a2,b):
    Y = a1 * X1 + a2 * X2 + b + np.random.normal(0,0.5,size=X1.shape)
    return Y

if __name__ == '__main__':
    
    print(np.random.normal(0,0.5,size=(4,)))
    exit()

    a1 = 0.5
    a2 = 0.2
    b = 2
    X1 = np.array([1,2,3,4])
    X2 = np.array([1,3,4,3])
    Y_label = fun(X1, X2, a1, a2, b)
    print(Y_label)
    

    X = np.linspace(0,5, 50)
    Y = np.linspace(0,5, 50)
    P,Q = np.meshgrid(X, Y)
    R = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            R[i,j] = a1 * X[i]  + a2 * Y[j] + b

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(P, Q, R)
    ax.scatter3D(X1,X2,Y_label, color='black')

    plt.show()
    """
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    import numpy as np

    ax = plt.axes(projection='3d')

    #三维线的数据
    zline = np.linspace(0, 15, 1000)
    xline = np.sin(zline)
    yline = np.cos(zline)
    ax.plot3D(xline, yline, zline, 'gray')

    # 三维散点的数据
    zdata = 15 * np.random.random(100)
    xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
    ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
    #ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    ax.scatter(xdata, ydata, zdata)
    plt.show()
    """