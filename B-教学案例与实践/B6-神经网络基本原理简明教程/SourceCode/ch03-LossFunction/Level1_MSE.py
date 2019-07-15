# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm

file_name = "../../data/ch03.npz"

def TargetFunction(x,w,b):
    y = w*x + b
    return y

def CreateSampleData(w,b,n):
    file = Path(file_name)
    if file.exists():
        data = np.load(file)
        x = data["data"]
        y = data["label"]
    else:
        x = np.linspace(0,1,num=n)
        noise = np.random.uniform(-0.5,0.5,size=(n))
        y = TargetFunction(x,w,b) + noise
        np.savez(file_name, data=x, label=y)
    #end if
    return x,y

def CostFunction(x,y,z,count):
    c = (z - y)**2
    loss = c.sum()/count/2
    return loss

def ShowResult(ax,x,y,a,loss,title):
    ax.scatter(x,y)
    ax.plot(x,a,'r')
    titles = str.format("{0} Loss={1:01f}",title,loss)
    ax.set_title(titles)

# 显示只变化b时loss的变化情况
def CalculateCostB(x,y,n,w,b):
    B = np.arange(b-1,b+1,0.05)
    Loss=[]
    for i in range(len(B)):
        z = w*x+B[i]
        loss = CostFunction(x,y,z,n)
        Loss.append(loss)
    plt.title("Loss according to b")
    plt.xlabel("b")
    plt.ylabel("J")
    plt.plot(B,Loss,'x')
    plt.show()

# 显示只变化w时loss的变化情况
def CalculateCostW(x,y,n,w,b):
    W = np.arange(w-1,w+1,0.05)
    Loss=[]
    for i in range(len(W)):
        z = W[i]*x+b
        loss = CostFunction(x,y,z,n)
        Loss.append(loss)
    plt.title("Loss according to w")
    plt.xlabel("w")
    plt.ylabel("J")
    plt.title = "Loss according to w"
    plt.plot(W,Loss,'o')
    plt.show()

# 显示同时变化w,b时loss的变化情况
def CalculateCostWB(x,y,n,w,b):
    W = np.arange(w-10,w+10,0.1)
    B = np.arange(b-10,b+10,0.1)
    Loss=np.zeros((len(W),len(B)))
    for i in range(len(W)):
        for j in range(len(B)):
            w = W[i]
            b = B[j]
            a = w*x+b
            loss = CostFunction(x,y,a,n)
            Loss[i,j] = loss

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(W,B,Loss)
    plt.show()
    

# 在一张图上分区域显示b的4种取值的loss情况
def show_cost_for_4b(x,y,n,w,b):
    fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2)
    a1 = w*x+b-1
    loss1 = CostFunction(x,y,a1,n)
    ShowResult(ax1,x,y,a1,loss1,"z=2x+2")
    a2 = w*x+b-0.5
    loss2 = CostFunction(x,y,a2,n)
    ShowResult(ax2,x,y,a2,loss2,"z=2x+2.5")
    a3 = w*x+b
    loss3 = CostFunction(x,y,a3,n)
    ShowResult(ax3,x,y,a3,loss3,"z=2x+3")
    a4 = w*x+b+0.5
    loss4 = CostFunction(x,y,a4,n)
    ShowResult(ax4,x,y,a4,loss4,"z=2x+3.5")
    plt.show()

# 在一张图上显示b的4种取值的比较
def show_all_4b(x,y,n,w,b):
    plt.scatter(x,y)
    z1 = w*x + b-1
    loss1 = CostFunction(x,y,z1,n)
    plt.plot(x,z1)

    z2 = w*x+b-0.5
    loss2 = CostFunction(x,y,z2,n)
    plt.plot(x,z2)

    z3 = w*x+b
    loss3 = CostFunction(x,y,z3,n)
    plt.plot(x,z3)

    z4 = w*x+b+0.5
    loss4 = CostFunction(x,y,z4,n)
    plt.plot(x,z4)
    plt.show()

        
def show_3d_surface(x,y,m,w,b):
    fig = plt.figure()
    ax = Axes3D(fig)

    X = x.reshape(m,1)
    Y = y.reshape(m,1)
    len1 = 50
    len2 = 50
    len = len1 * len2
    W = np.linspace(w-2, w+2, len1)
    B = np.linspace(b-2, b+2, len2)
    W, B = np.meshgrid(W, B)

    m = X.shape[0]
    Z = np.dot(X, W.ravel().reshape(1,len)) + B.ravel().reshape(1,len)
    Loss1 = (Z - Y)**2
    Loss2 = Loss1.sum(axis=0,keepdims=True)/m/2
    Loss3 = Loss2.reshape(len1, len2)
    ax.plot_surface(W, B, Loss3, norm=LogNorm(), cmap='rainbow')
    plt.show()

def test_2d(x,y,m,w,b):
    s = 200
    W = np.linspace(w-2,w+2,s)
    B = np.linspace(b-2,b+2,s)
    LOSS = np.zeros((s,s))
    for i in range(len(W)):
        for j in range(len(B)):
            z = W[i] * x + B[j]
            loss = CostFunction(x,y,z,m)
            LOSS[i,j] = round(loss, 2)
    print(LOSS)
    print("please wait for 20 seconds...")
    while(True):
        X = []
        Y = []
        is_first = True
        loss = 0
        for i in range(len(W)):
            for j in range(len(B)):
                if LOSS[i,j] != 0:
                    if is_first:
                        loss = LOSS[i,j]
                        X.append(W[i])
                        Y.append(B[j])
                        LOSS[i,j] = 0
                        is_first = False
                    elif (LOSS[i,j] == loss) or (abs(loss / LOSS[i,j] -  1) < 0.02):
                        X.append(W[i])
                        Y.append(B[j])
                        LOSS[i,j] = 0
        if is_first == True:
            break
        plt.plot(X,Y,'.')
    
    plt.xlabel("w")
    plt.ylabel("b")
    plt.show()

def draw_contour(x,y,m,w,b):
    X = x.reshape(m,1)
    Y = y.reshape(m,1)
    len1 = 50
    len2 = 50
    len = len1 * len2
    W = np.linspace(w-2, w+2, len1)
    B = np.linspace(b-2, b+2, len2)
    W, B = np.meshgrid(W, B)
    LOSS = np.zeros((len1, len2))

    m = X.shape[0]
    Z = np.dot(X, W.ravel().reshape(1,len)) + B.ravel().reshape(1,len)
    Loss1 = (Z - Y)**2
    Loss2 = Loss1.sum(axis=0,keepdims=True)/m/2
    Loss3 = Loss2.reshape(len1, len2)
    plt.contour(W,B,Loss3,levels=np.logspace(-5, 5, 50), norm=LogNorm(), cmap=plt.cm.jet)
    plt.show()

if __name__ == '__main__':
    
    m=50
    w=2
    b=3
    x,y=CreateSampleData(w,b,m)
    plt.scatter(x,y)
    #plt.axis([0,1.1,0,4.2])
    plt.show()
    
    show_cost_for_4b(x,y,m,w,b)
    show_all_4b(x,y,m,w,b)

    CalculateCostB(x,y,m,w,b)
    CalculateCostW(x,y,m,w,b)
    
    #CalculateCostWB(x,y,n)

    show_3d_surface(x,y,m,w,b)
    
    draw_contour(x,y,m,w,b)

    test_2d(x,y,m,w,b)
