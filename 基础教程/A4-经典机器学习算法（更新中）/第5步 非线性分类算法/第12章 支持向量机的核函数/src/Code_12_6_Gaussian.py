
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from scipy import integrate

def gaussian_1d(x, sigma, mu):
    a = - (x-mu)**2 / (2*sigma*sigma)
    f = np.exp(a) / (sigma * np.sqrt(2*np.pi))
    return f

# 验证fa·fb的积分值
def f(x):
    sigma=1
    # 式 11.9.8
    f = np.exp(-(2*x*x + 2*x + 5)/(2*sigma*sigma)) / (2*np.pi*sigma*sigma)
    return f

def integrate_fab():
    return integrate.quad(f,-100,100)

# 高斯函数图像
# sample - 样本点
# gamma - 控制形状
# weight - 控制权重（正负及整体升高或降低）
# scope - 网格数据范围及密度
def gaussian_2d_fun(sample, gamma, scope):
    # 生成网格数据
    xx = np.linspace(scope[0], scope[1], scope[2])
    yy = np.linspace(scope[3], scope[4], scope[5])
    P,Q = np.meshgrid(xx, yy)
    # 二维高斯函数 * 权重 -> 三维高斯曲面
    R = np.exp(-gamma * ((P-sample[0])**2 + (Q-sample[1])**2))
    return P,Q,R

# 绘图区基本设置
def set_ax(ax, scope):
    if (ax.name != '3d'):
        ax.grid()
        ax.axis('equal')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    if (scope is not None):
        ax.set_xlim(scope[0], scope[1])
        ax.set_ylim(scope[3], scope[4])


if __name__ == '__main__':

    print("积分结果：")
    print(integrate_fab())

    x = np.linspace(-5,5,100)

    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.title(u"一维高斯函数与其内积（相关性）的理解")
    plt.axis('off')

    # 1. 基本高斯函数（二维）
    ax1 = fig.add_subplot(121)
    sigma = [1,1.5,1.5,3]
    mu = [0,0,2,0]
    linestyles=['-', '-.', ':', '--']
    for i in range(4):
        f = gaussian_1d(x, sigma[i], mu[i])
        label = str.format("μ={0},σ={1}", mu[i], sigma[i])
        ax1.plot(x, f, linestyle=linestyles[i], label=label)
    ax1.legend()
    ax1.grid()

    # 2. 两个高斯函数的乘积，表示相关性
    sigma = 1
    x = np.linspace(-5,5,50)
    fa = gaussian_1d(x, sigma, 1)
    fb = gaussian_1d(x, sigma, -2)
    fab = fa * fb
    
    ax2 = fig.add_subplot(122)
    ax2.plot(x, fa, linestyle='--', marker='.', label='fa(μ=1,σ=1)')
    ax2.plot(x, fb, linestyle=':', marker='*', label='fb(μ=-2,σ=1)')
    ax2.plot(x, fab, label='fa·fb')

    ax2.grid()
    ax2.legend()

    plt.show()


    fig = plt.figure()
    plt.title(u"二维高斯函数与其内积（相关性）的理解")
    plt.axis('off')
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)   
    scope = [-1.5,2.5,50, -1.5,2.5,50]
    set_ax(ax1, scope)
    set_ax(ax2, scope)
    gamma = 1
    P,Q,R1 = gaussian_2d_fun([0,1], gamma, scope)
    ax1.plot_surface(P, Q, R1, cmap=cm.coolwarm)
    ax2.contour(P, Q, R1, colors='red', alpha=0.2, linewidth=0.5) #cmap=cm.coolwarm)

    P,Q,R2 = gaussian_2d_fun([1,0], gamma, scope)
    ax1.plot_surface(P, Q, R2, cmap=cm.coolwarm)
    ax2.contour(P, Q, R2, colors='blue', alpha=0.2, linewidth=0.5) #cmap=cm.coolwarm)
    
    R = R1 * R2
    ax1.plot_surface(P, Q, R, cmap=cm.coolwarm)
    ax2.contour(P, Q, R, colors='green') # cmap=cm.coolwarm)


    plt.show()
