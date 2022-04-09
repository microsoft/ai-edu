import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    y = 1/(x*x)
    return y

def f2(x):
    y = np.sin(x)
    return y

def f3(x):
    y = 0.4 * x * x + 0.3 * x * np.sin(15*x) + 0.01 * np.cos(50*x) - 0.3
    return y

def integral(f, a, b, n):
    v = 0
    repeat = 10
    for i in range(repeat):
        x = np.random.uniform(a, b, size=(n, 1))
        y = f(x)
        v += np.sum(y) / n * (b-a)
    return v/repeat

def show(ax, f, a, b, n, v):
    # 绘制函数曲线
    x = np.linspace(a, b, n)
    y = f(x)
    ax.set_title("integral="+str.format("{0:.2f}",v))
    ax.grid()
    ax.plot(x, y, c='g')

    y_min = np.min(y)
    y_max = np.max(y)
    X = np.random.uniform(a, b, n)
    Y = np.random.uniform(y_min, y_max, n)
    for x,y in zip(X,Y):
        if y < f(x):
            ax.plot(x,y,'o',markersize=1,c='r')
        else:
            ax.plot(x,y,'o',markersize=1,c='b')

if __name__=="__main__":
    v1 = integral(f1, 0.2, 1, 10000)
    print("S1 =",v1)
    v2 = integral(f2, 0, 3.1416, 10000)
    print("S2 =",v2)
    v3 = integral(f3, 0, 1, 10000)
    print("S3 =",v3)

    fig = plt.figure()
    ax = fig.add_subplot(131)
    show(ax, f1, 0.2, 1, 100, v1)
    ax = fig.add_subplot(132)
    show(ax, f2, 0, 3.1416, 100, v2)
    ax = fig.add_subplot(133)
    show(ax, f3, 0, 1, 100, v3)
    
    plt.show()