import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def draw_left_3d(fig):
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title(u"约束平面与曲面相交")

    # 绘制 x^2+y^2 曲面
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    P, Q = np.meshgrid(x, y)
    R = P * P + Q * Q
    ax.plot_surface(P, Q, R, alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # 绘制x+y+2=0平面，垂直于x-y 平面
    x = np.linspace(-3, 1, 100)
    y = np.linspace(-3, 1, 100)
    X, Y = np.meshgrid(x, y)
    P2 = X
    Q2 = -X-2
    R2 = 20*np.maximum(Y,0) # 增加平面的高度
    ax.plot_surface(P2, Q2, R2, color='y', alpha=0.6)

    #两平面交线
    x = np.linspace(-3, 1, 100)
    y = -x - 2
    z = 2 * np.square(x) + 4 * x + 4
    ax.plot3D(x, y, z, color='r', alpha=0.6)

    # 极值点
    ax.scatter(-1, -1, 2, c='r')


def draw_right_2d(fig):
    ax2 = fig.add_subplot(122)
    ax2.grid()
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(u"约束直线与投影等高线相切")

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    P, Q = np.meshgrid(x, y)
    R = P * P + Q * Q
    # 绘制等高线
    c = ax2.contour(P, Q, R, levels=np.linspace(-10, 10, 11))
    ax2.clabel(c,inline=1,fontsize=10)
    # 绘制约束直线
    x = np.linspace(-3, 1, 100)
    y = -x - 2
    ax2.plot(x, y)
    ax2.scatter(-1,-1,c='r')


if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()

    draw_left_3d(fig)
    draw_right_2d(fig)

    plt.show()

