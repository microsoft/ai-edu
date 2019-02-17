import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl

def TargetFunction(x):
    y = 3*x + 1
    return y

def CreateSampleData(n):
    x = np.linspace(0,1,num=n)
    noise = np.random.uniform(-0.5,0.5,size=(n))
    print(noise)
    y = TargetFunction(x) + noise
    return x,y

def CostFunction(x,y,a,count):
    c = (a - y)**2
    loss = c.sum()/count/2
    return loss

def ShowResult(ax,x,y,a,loss,title):
    ax.scatter(x,y)
    ax.plot(x,a,'r')
    titles = str.format("{0} Loss={1:01f}",title,loss)
    ax.set_title(titles)

# 显示只变化b时loss的变化情况
def CalculateCostB(x,y,n):
    B = np.arange(0,2,0.05)
    Loss=[]
    for i in range(len(B)):
        b = B[i]
        a = 3*x+b
        loss = CostFunction(x,y,a,n)
        Loss.append(loss)
    plt.title("Loss according to b")
    plt.xlabel("b")
    plt.ylabel("Loss")
    plt.plot(B,Loss,'x')
    plt.show()

# 显示只变化w时loss的变化情况
def CalculateCostW(x,y,n):
    W = np.arange(2,4,0.05)
    Loss=[]
    for i in range(len(W)):
        w = W[i]
        a = w*x+1
        loss = CostFunction(x,y,a,n)
        Loss.append(loss)
    plt.title("Loss according to w")
    plt.xlabel("w")
    plt.ylabel("Loss")
    plt.title = "Loss according to w"
    plt.plot(W,Loss,'o')
    plt.show()

# 显示同时变化w,b时loss的变化情况
def CalculateCostWB(x,y,n):
    W = np.arange(-7,13,0.1)
    B = np.arange(-9,11,0.1)
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
def show_cost_for_4b(x,y,n):
    fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2)
    a1 = 3*x
    loss1 = CostFunction(x,y,a1,n)
    ShowResult(ax1,x,y,a1,loss1,"y=3x")
    a2 = 3*x+0.5
    loss2 = CostFunction(x,y,a2,n)
    ShowResult(ax2,x,y,a2,loss2,"y=3x+0.5")
    a3 = 3*x+1
    loss3 = CostFunction(x,y,a3,n)
    ShowResult(ax3,x,y,a3,loss3,"y=3x+1")
    a4 = 3*x+1.5
    loss4 = CostFunction(x,y,a4,n)
    ShowResult(ax4,x,y,a4,loss4,"y=3x+1.5")
    plt.show()

# 在一张图上显示b的4种取值的比较
def show_all_4b(x,y,n):
    plt.scatter(x,y)
    plt.axis([0,1.1,0,4.2])
    a1 = 3*x
    loss1 = CostFunction(x,y,a1,n)
    plt.plot(x,a1)

    a2 = 3*x+0.5
    loss2 = CostFunction(x,y,a2,n)
    plt.plot(x,a2)

    a3 = 3*x+1
    loss3 = CostFunction(x,y,a3,n)
    plt.plot(x,a3)

    a4 = 3*x+1.5
    loss4 = CostFunction(x,y,a4,n)
    plt.plot(x,a4)
    plt.show()

def test():
    x=np.arange(-10,10,step=0.1)
    y=np.arange(-10,10,step=0.1)
    z=np.zeros((200,200))
    for i in range(200):
        for j in range(200):
            z[i,j] = (x[i]+y[j])**2

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.plot_surface(x,y,z)
    ax.plot_wireframe(x,y,z)
    plt.show()
        
  
n=100
x,y=CreateSampleData(n)
plt.scatter(x,y)
plt.axis([0,1.1,0,4.2])
plt.show()

show_cost_for_4b(x,y,n)
show_all_4b(x,y,n)

CalculateCostB(x,y,n)
CalculateCostW(x,y,n)

#CalculateCostWB(x,y,n)
#test()

