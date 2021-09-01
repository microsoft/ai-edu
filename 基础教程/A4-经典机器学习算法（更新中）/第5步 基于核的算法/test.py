
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.core.fromnumeric import argmin

# 统计学习方法中的例子图示
def test():

    #a1 = np.linspace(-2,2,1000)
    #a2 = np.linspace(-2,2,1000)
    a1 = np.linspace(0,0.3,1000)
    a2 = np.linspace(0,0.3,1000)

    P,Q = np.meshgrid(a1, a2)
    R = 4*P*P + 13/2*Q*Q+10*P*Q-2*P-2*Q
    z = np.min(R)
    idx = np.argmin(R)
    y = a1[idx // 1000]
    x = a2[idx % 1000]

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    ax.scatter(x,y,z)
    print(x,y,z)

    ax.plot_surface(P, Q, R, alpha=0.5)
    ax.set_xlabel("a1")
    ax.set_ylabel("a2")
    plt.show()


# 拉格朗日
def test2():
    x = np.linspace(-3,3,100)
    y1 = x*x
    y2 = 0.5*x+1
    plt.plot(x,y1)
    plt.plot(x,y2)
    plt.grid()
    plt.show()

def test3():
    x = np.linspace(-3,3,500)
    y = np.linspace(-3,3,500)

    P,Q = np.meshgrid(x, y)
    R = P*P + Q*Q
    """
    z = np.min(R)
    idx = np.argmin(R)
    y = a1[idx // 1000]
    x = a2[idx % 1000]
    """
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    ax.plot_surface(P, Q, R, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    R2 = P+1.5*Q+3
    ax.plot_surface(P, Q, R2, alpha=0.5)

    x = []
    y = []
    r = []
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if (abs(R[i,j]-R2[i,j]) <= 0.1):
                r.append(R[i,j])
                x.append(i)
                y.append(j)

    x = np.array(x)
    x = (x-250)*6/500
    y = np.array(y)
    y = (y-250)*6/500
    ax.scatter(y,x,r,s=1,color='b')

    print(min(r))
    idx = argmin(r)
    print(x[idx], y[idx])
    ax.scatter(y[idx],x[idx],min(r),s=10,color='r')


    x = -12/13
    y = -18/13
    z = x + 1.5*y + 3
    z2 = x*x+y*y
    print(x,y,z,z2)
    ax.scatter(x,y,z, color='r')
    ax.scatter(x,y,z2, color='r')

    """
    x = 1.366
    y = 1.366
    z1 = x*x + y*y
    z2 = x + y + 1
    ax.scatter(x,y,z1)
    ax.scatter(x,y,z2)
    print(z1,z2)
    """
    plt.show()




if __name__ == '__main__':
    test3()
