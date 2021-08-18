import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 图1，y限制在-2，2之间，然后随机加一些权重，放3个图
def show_result():
    x = np.linspace(-2,2,100)
    x2 = x * x / 2
    x3 = x * x * x / 3
    x4 = x * x * x * x / 4
    x5 = x * x * x * x * x / 5
    plt.plot(x,x)
    plt.plot(x,x2)
    plt.plot(x,x3)
    plt.plot(x,x4)
    plt.plot(x,x5)
    plt.show()

if __name__ == '__main__':

    show_result()


