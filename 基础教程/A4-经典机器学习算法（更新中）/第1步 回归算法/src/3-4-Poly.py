import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 图1，y限制在-2，2之间，然后随机加一些权重，放3个图
def show_result():
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.title(u"多项式回归的原理")
    plt.axis('off')

    # 绘制左视图
    ax = fig.add_subplot(121)
    ax.grid()
    ax.set_xlabel(u"x 的多次幂函数的图像")
    x = np.linspace(-2,2,20)
    x2 = x * x /2
    x3 = x * x * x /5
    x4 = x * x * x * x /10
    x5 = x * x * x * x * x /20
    ax.plot(x,x, label=r'$x$', marker='o', linestyle='-')
    ax.plot(x,x2, label=r'$x^2$', linestyle='--')
    ax.plot(x,x3, label=r'$x^3$', linestyle='-.')
    ax.plot(x,x4, label=r'$x^4$', linestyle=':')
    ax.plot(x,x5, label=r'$x^5$', linestyle='-')
    ax.legend()

    # 右子图
    ax = fig.add_subplot(122)
    ax.grid()
    ax.set_xlabel(u"x 的多项式的随机参数组合")
    ls = ["-", "--", "-.", ":", "-"]
    for i in range(5):
        A = (np.random.random((6,1)) - 0.5)  * 2
        ones = np.ones((20,1))
        X = np.column_stack((ones, x,x2,x3,x4,x5))
        Y = np.dot(X, A)
        ax.plot(x, Y, linestyle=ls[i], label = str(i+1))
        print(str.format("{0}: {1}", i+1, A))
    ax.legend()

    plt.xlim((-2,2))
    plt.ylim((-2,2))
    
    plt.show()


if __name__ == '__main__':

    show_result()


