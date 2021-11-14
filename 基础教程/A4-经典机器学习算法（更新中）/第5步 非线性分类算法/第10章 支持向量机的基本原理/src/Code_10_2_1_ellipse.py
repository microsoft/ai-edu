import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def draw_left_3d(fig):
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title(u"约束平面与曲面相交")

    # 绘制 x^2+2y^2 曲面
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    P, Q = np.meshgrid(x, y)
    R = P * P + Q * Q * 2
    ax.plot_surface(P, Q, R, alpha=0.5)
    ax.contour(P, Q, R, [0.5,1.5,2.67,3.5,4.5], zdir='z', offset=0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # 绘制x+y+2=0平面，垂直于x-y 平面
    x = np.linspace(-3, 1, 100)
    y = np.linspace(-3, 1, 100)
    X, Y = np.meshgrid(x, y)
    P2 = X
    Q2 = -X-2
    R2 = 15*np.maximum(Y,0) # 增加平面的高度
    ax.plot_surface(P2, Q2, R2, color='y', alpha=0.6)

    #两平面交线
    x = np.linspace(-3, 1, 100)
    y = -x - 2
    z = 3 * x * x + 8 * x + 8
    ax.plot3D(x, y, z, color='r', alpha=0.6)
    print(np.min(z), np.argmin(z))
    print(x[np.argmin(z)])

    # 极值点
    ax.scatter(-4/3, -2/3, 24/9, c='r')

    return P,Q,R


def draw_right_2d(fig, P, Q, R):
    ax2 = fig.add_subplot(122)
    ax2.grid()
    ax2.axis('equal')
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(u"约束直线与投影等高线相切")

    # 绘制等高线
    c = ax2.contour(P, Q, R, [0.5,1.5,2.67,3.5,4.5])
    ax2.clabel(c,inline=1,fontsize=10)
    # 绘制约束直线
    x = np.linspace(-3, 1, 100)
    y = -x - 2
    ax2.plot(x, y)
    ax2.scatter(-4/3, -2/3, c='r')

def f(x):
    return x**4+6*x**3+17*x**2-12*x-16

def aaa():
    X = np.linspace(1.0841, 1.0842, 100)
    print(X)
    for x in X:
        print(x, f(x))


if __name__ == '__main__':

    aaa()
    exit()

    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()

    P,Q,R = draw_left_3d(fig)
    draw_right_2d(fig, P, Q, R)

    plt.show()

