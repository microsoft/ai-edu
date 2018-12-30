import numpy as np
import matplotlib.pyplot as plt

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
  
n=12
x,y=CreateSampleData(n)
plt.scatter(x,y)
plt.axis([0,1.1,0,4.2])
plt.show()

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
