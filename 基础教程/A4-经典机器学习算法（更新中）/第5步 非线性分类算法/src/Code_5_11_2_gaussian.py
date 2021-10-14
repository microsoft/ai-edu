
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from scipy import integrate

def gaussian_2d(x, sigma, mu):
    a = - (x-mu)**2 / (2*sigma*sigma)
    f = np.exp(a) / (sigma * np.sqrt(2*np.pi))
    return f

'''
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
'''

# 验证fa·fb的积分值
def f(x):
    sigma=1
    f = np.exp(-(2*x*x + 2*x + 5)/(2*sigma*sigma)) / (2*np.pi*sigma*sigma)
    return f

def integrate_fab():
    return integrate.quad(f,-100,100)


if __name__ == '__main__':

    print("积分结果：")
    print(integrate_fab())

    x = np.linspace(-5,5,100)

    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.title(u"高斯函数与其内积（相关性）的理解")
    plt.axis('off')

    # 1. 基本高斯函数（二维）
    ax1 = fig.add_subplot(121)
    sigma = [1,1.5,1.5,3]
    mu = [0,0,2,0]
    linestyles=['-', '-.', ':', '--']
    for i in range(4):
        f = gaussian_2d(x, sigma[i], mu[i])
        label = str.format("μ={0},σ={1}", mu[i], sigma[i])
        ax1.plot(x, f, linestyle=linestyles[i], label=label)
    ax1.legend()
    ax1.grid()

    # 2. 两个高斯函数的乘积，表示相关性
    sigma = 1
    x = np.linspace(-5,5,50)
    #f1 = gaussian_2d(x, sigma, 0)
    fa = gaussian_2d(x, sigma, 1)
    fb = gaussian_2d(x, sigma, -2)

    #f12 = f1 * f2
    #f13 = f1 * f3
    fab = fa * fb
    
    ax2 = fig.add_subplot(122)
    #ax2.plot(x, f1, label='f1(μ=0,σ=1)')
    ax2.plot(x, fa, linestyle='--', marker='.', label='fa(μ=1,σ=1)')
    ax2.plot(x, fb, linestyle=':', marker='*', label='fb(μ=-2,σ=1)')
    #ax2.plot(x, f12, linestyle='--', marker='^', label='f1*f2')
    #ax2.plot(x, f13, linestyle=':', marker='*', label='f1*f3')
    ax2.plot(x, fab, label='fa·fb')

    ax2.grid()
    ax2.legend()

    plt.show()

