import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 图1，y限制在-2，2之间，然后随机加一些权重，放3个图
def show_result():
    x = np.linspace(-2,2,100)
    x2 = x * x /2
    x3 = x * x * x /5
    x4 = x * x * x * x /10
    x5 = x * x * x * x * x /20
    plt.plot(x,x)
    plt.plot(x,x2)
    plt.plot(x,x3)
    plt.plot(x,x4)
    plt.plot(x,x5)

    A = (np.random.random((6,1)) - 0.5) 
    ones = np.ones((100,1))
    X = np.column_stack((ones, x,x2,x3,x4,x5))
    Y = np.dot(X, A)
    plt.plot(x, Y, color='Red')


    plt.xlim((-2,2))
    plt.ylim((-2,2))
    plt.grid()
    plt.show()


if __name__ == '__main__':

    show_result()


