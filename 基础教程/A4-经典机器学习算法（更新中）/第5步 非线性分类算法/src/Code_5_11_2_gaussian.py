
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

def gaussian_2d(x, sigma, mu):
    a = - (x-mu)**2 / (2*sigma*sigma)
    f = np.exp(a) / (sigma * np.sqrt(2*np.pi))
    return f

def gaussian_3d(ax):
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
    plt.title(u"高斯函数与相关性的理解")
    plt.axis('off')

    # 1. 基本高斯函数（二维）
    ax1 = fig.add_subplot(131)
    sigma = [1,1.5,1.5,3]
    mu = [0,0,2,0]
    linestyles=['-', '-.', ':', '--']
    for i in range(4):
        f = gaussian_2d(x, sigma[i], mu[i])
        ax1.plot(x, f, linestyle=linestyles[i], label='μ=0,σ=1')
    ax1.legend()
    ax1.grid()

    # 2. 两个高斯函数的乘积，表示相关性
    sigma = 1
    x = np.linspace(-5,5,50)
    f1 = gaussian_2d(x, sigma, 0)
    f2 = gaussian_2d(x, sigma, 1)
    f3 = gaussian_2d(x, sigma, -2)

    f12 = f1 * f2
    f13 = f1 * f3
    
    ax2 = fig.add_subplot(132)
    ax2.plot(x, f1, label='f1(μ=0,σ=1)')
    ax2.plot(x, f2, linestyle='--', label='f2(μ=1,σ=1)')
    ax2.plot(x, f3, linestyle=':', label='f3(μ=-2,σ=1)')
    ax2.plot(x, f12, linestyle='--', marker='.', label='f1*f2')
    ax2.plot(x, f13, linestyle=':', marker='*', label='f1*f3')

    ax2.grid()
    ax2.legend()

    # 3. 三维高斯函数

    ax3 = fig.add_subplot(133, projection='3d')
    gaussian_3d(ax3)

    plt.show()