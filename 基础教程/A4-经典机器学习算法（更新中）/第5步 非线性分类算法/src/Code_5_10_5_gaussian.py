
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

def gaussian_fun(x, sigma, mu):
    a = - (x-mu)**2 / (2*sigma*sigma)
    f = np.exp(a) / (sigma * np.sqrt(2*np.pi))
    return f

def gaussian_fun2(ax):
    xx = np.linspace(-5,5,100)
    yy = np.linspace(-5,5,100)
    P,Q = np.meshgrid(xx, yy)
    R1 = np.exp(-(P*P + Q*Q))
    #ax.plot_surface(P, Q, R1, cmap=cm.coolwarm)
    R2 = np.exp(-(P-1)**2/2 - (Q-2)**2/2)
    R = R1 + R2
    #ax.plot_surface(P, Q, R, cmap=cm.coolwarm)

    cset = ax.contour(P,Q,R,zdir='z',offset=0,cmap=cm.coolwarm)                     #绘制xy面投影
    cset = ax.contour(P,Q,R,zdir='x',offset=-4,cmap = mpl.cm.winter)                 #绘制zy面投影
    cset = ax.contour(P,Q,R,zdir='y',offset= 4,cmap =mpl.cm.winter)                  #绘制zx面投影

if __name__ == '__main__':
    x = np.linspace(-5,5,100)

    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.title(u"高斯函数参数理解")
    plt.axis('off')

    ax1 = fig.add_subplot(121)

    f = gaussian_fun(x, 1, 0)
    ax1.plot(x, f, linestyle='-', label='μ=0,σ=1')

    f = gaussian_fun(x, 1.5, 0)
    ax1.plot(x,f,linestyle='-.',label='μ=0,σ=1.5')

    f = gaussian_fun(x, 1.5, 2)
    ax1.plot(x,f,linestyle=':',label='μ=2,σ=1.5')

    f = gaussian_fun(x, 3, 0)
    ax1.plot(x,f,linestyle='--',label='μ=0,σ=3')

    ax1.legend()
    ax1.grid()

    ax2 = fig.add_subplot(122, projection='3d')
    gaussian_fun2(ax2)

    plt.show()