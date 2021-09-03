
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
    x = np.linspace(-3,3,1000)
    y = np.linspace(-3,3,1000)

    P,Q = np.meshgrid(x, y)
    R = P*P + Q*Q
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    ax.plot_surface(P, Q, R, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    """
    P,Q = np.meshgrid(x, y)
    R2 = np.zeros_like(P)
    for i in range(P.shape[0]*P.shape[1]):
        for j in range(Q.shape[0]*Q.shape[1]):
            if (P[i]+Q[j]+2==0):
                R2[i,j]=1
    ax.plot_surface(P, Q, R2, alpha=0.5)
    """
    
    #for z in np.linspace(0,20,100):
    #    ax.plot([-3,1],[1,-3],z)
    
    x = -1
    y = -1
    z = 2
    ax.scatter(x,y,z,c='r')

    """
    x = []
    y = []
    r = []
    min_r = 10
    min_i = 0
    min_j=0
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if (abs(R[i,j]-R2[i,j]) <= 0.1):
                r.append(R[i,j])
                x.append(i)
                y.append(j)
                if (R[i,j] < min_r):
                    min_r = R[i,j]
                    min_i = i
                    min_j = j

    print(min_r, min_i, min_j)

    x = np.array(x)
    x = (x-500)*6/1000
    y = np.array(y)
    y = (y-500)*6/1000
    ax.scatter(x,y,r,s=1,color='b')

    print("min r = ",min(r))
    idx = argmin(r)
    print("idx=", idx)
    print(x[idx], y[idx])
    ax.scatter(x[idx],y[idx],min(r),s=10,color='r')


    """
  
    plt.show()




if __name__ == '__main__':
    test3()
