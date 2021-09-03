from matplotlib.colors import LogNorm
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

    R2 = P + Q + 2
    ax.plot_surface(P, Q, R2, alpha=0.5)
  
    #x*x + y*y -x-y-2=0
    #ax.plot3D(x,y,z)

    
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
    ax.scatter(y,x,r,s=1,color='b')

    print("min r = ",min(r))
    idx = argmin(r)
    print("idx=", idx)
    print(x[idx], y[idx])
    ax.scatter(x[idx],y[idx],min(r),s=10,color='r')
    print("max r = ",max(r))
    idx = np.argmax(np.array(r))
    print("idx=", idx)
    print(x[idx], y[idx])
    ax.scatter(x[idx],y[idx],max(r),s=10,color='r')
    
  
    plt.show()



def test4():
    x = np.linspace(-3, 3, 1000)
    y = np.linspace(-3, 3, 1000)
    P, Q = np.meshgrid(x, y)
    R = P * P + Q * Q
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(P, Q, R, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # 二元函数定义域平面
    x = np.linspace(-3, 2, 1000)
    y = np.linspace(-3, 2, 1000)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X,
                    Y=-X * 1 - 2,
                    Z=8*np.maximum(Y,0),
                    color='y',
                    alpha=0.6
                    )

    # for z in np.linspace(0,20,100):
    ax.plot([-3, 1], [1, -3])
    x = -1
    y = -1
    z = 2
    ax.scatter(x, y, z, c='r')
    #两平面交线
    x = np.linspace(-3, 1, 1000)
    y = -x - 2
    z = 2 * np.square(x) + 4 * x + 4
    ax.plot3D(x,
              y,
              z,
              color='b',
              alpha=0.6
              )
    
    plt.show()


def test5():
    x = np.linspace(-3, 3, 1000)
    y = np.linspace(-3, 3, 1000)
    P, Q = np.meshgrid(x, y)
    R = P * P + Q * Q
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(P, Q, R, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    R2 = P + Q + 2
    ax.plot_surface(P, Q, R2, alpha=0.5)
    #    x*x + y*y -x-y-2=0
    #   ax.plot3D(x,y,z)
    x = []
    y = []
    r = []
    min_r = 10
    min_i = 0
    min_j = 0
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if (abs(R[i, j] - R2[i, j]) <= 0.01):
                r.append(R[i, j])
                x.append(i)
                y.append(j)
                if (R[i, j] < min_r):
                    min_r = R[i, j]
                    min_i = i
                    min_j = j
    print("xiaowu:", min_r, min_i, min_j)
    x = np.array(x)
    x = (x - 500) * 6 / 1000
    y = np.array(y)
    y = (y - 500) * 6 / 1000
    #print(x[min_i], y[min_j])
    #ax.scatter(x, y, r, s=1, color='b')
    
    print("min r = ", min(r))
    idx = argmin(r)
    print("idx=", idx)
    print(x[idx], y[idx])
    ax.scatter(x[idx], y[idx], min(r), s=10, color='r')
    print("max r = ", max(r))
    idx = np.argmax(np.array(r))
    print("idx=", idx)
    print(x[idx], y[idx])
    ax.scatter(x[idx], y[idx], max(r), s=10, color='r')
    
    t = np.linspace(0, 10, 2000)
    a = np.cos(t) + np.sin(t)
    x = ((a + np.sqrt(np.square(a) + 8)) / 2) * np.cos(t)
    y = ((a + np.sqrt(np.square(a) + 8)) / 2) * np.sin(t)
    z = np.square(x)+np.square(y)
    print("minz=",np.min(z))
    idx = np.argmin(z)
    print(x[idx],y[idx])
    ax.plot3D(x,
              y,
              z,
              color='r',
              alpha=0.6
              )
    plt.show()



def test6():
    x = np.linspace(-3, 3, 1000)
    y = np.linspace(-3, 3, 1000)
    P, Q = np.meshgrid(x, y)
    R = P * P + Q * Q
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(P, Q, R, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.contour(P,Q,R,levels=np.logspace(-5, 5, 50), norm=LogNorm(), cmap=plt.cm.jet)

    R2 = P*P*Q-3
    ax.plot_surface(P, Q, R2, alpha=0.5)

    plt.contour(P,Q,R2,levels=np.logspace(-5, 5, 50), norm=LogNorm(), cmap=plt.cm.jet)

    x=1.618
    y=1.145
    ax.scatter(x,y,x*x+y*y,s=20,color='r')
    ax.scatter(x,y,x*x*y-3,s=20,color='b')

    x = []
    y = []
    r = []
    min_r = 10
    min_i = 0
    min_j = 0
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if (abs(R[i, j] - R2[i, j]) <= 0.01):
                r.append(R[i, j])
                x.append(i)
                y.append(j)
                if (R[i, j] < min_r):
                    min_r = R[i, j]
                    min_i = i
                    min_j = j
    print("xiaowu:", min_r, min_i, min_j)
    x = np.array(x)
    x = (x - 500) * 6 / 1000
    y = np.array(y)
    y = (y - 500) * 6 / 1000
    #print(x[min_i], y[min_j])
    ax.scatter(y, x, r, s=1, color='b')


    plt.show()

if __name__ == '__main__':
    test6()
