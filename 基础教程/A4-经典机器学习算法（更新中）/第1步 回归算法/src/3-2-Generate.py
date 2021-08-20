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
    ax.set_xlabel(u"x 的多次幂函数的图像")
    x = np.linspace(-2,2,100)
    x2 = x * x /2
    x3 = x * x * x /5
    x4 = x * x * x * x /10
    x5 = x * x * x * x * x /20
    ax.plot(x,x)
    ax.plot(x,x2)
    ax.plot(x,x3)
    ax.plot(x,x4)
    ax.plot(x,x5)

    ax = fig.add_subplot(122)
    ax.set_xlabel(u"x 的多次幂函数的图像")
    for i in range(5):
        A = (np.random.random((6,1)) - 0.5)  * 2
        ones = np.ones((100,1))
        X = np.column_stack((ones, x,x2,x3,x4,x5))
        Y = np.dot(X, A)
        plt.plot(x, Y)


    plt.xlim((-2,2))
    plt.ylim((-2,2))
    plt.grid()
    plt.show()


if __name__ == '__main__':

    show_result()


