import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle

def hh(x,y):
    return (x+2)**2+(y+1)**2-1

def gg(x,y):
    return x**2+2*y**2

def draw_left_3d(ax):

    ax.set_title(u"约束平面与曲面相交")

    # 绘制 x^2+2y^2 曲面
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    P, Q = np.meshgrid(x, y)
    R = P * P + Q * Q * 2
    ax.plot_surface(P, Q, R, alpha=0.5)
    #ax.contour(P, Q, R, [0,0.5,1,1.5,1.7579,2,2.5,3,4], zdir='z', offset=0)
    ax.contour(P, Q, R, [1,1.7579,2.5])
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    for xx in x:
        for yy in y:
            if (abs(hh(xx,yy)) <= 0.05):
                ax.scatter(xx,yy,gg(xx,yy),color='r',s=5)

    #return P,Q,R

    # 绘制约束圆
    #return P,Q,R
    # 生成圆柱数据，底面半径为r，高度为h。
    # 先根据极坐标方式生成数据
    u = np.linspace(0,2*np.pi,50)  # 把圆分按角度为50等分
    h = np.linspace(0,20,2)        # 把高度1均分为20份
    x = np.outer(np.sin(u),np.ones(len(h)))-2  # x值重复20次
    y = np.outer(np.cos(u),np.ones(len(h)))-1  # y值重复20次
    z = np.outer(np.ones(len(u)),h)   # x，y 对应的高度

    # Plot the surface
    ax.plot_surface(x, y, z, cmap=plt.get_cmap('rainbow'))



    return P,Q,R



import math

def draw_right_2d2(ax2, P, Q, R):
    ax2.grid()
    ax2.axis('equal')
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(u"约束直线与投影等高线相切")

    # 绘制约束圆
    circle = Circle((-2,-1), radius=1, facecolor='white', edgecolor='red')
    ax2.add_patch(circle)

    # 绘制等高线
    c = ax2.contour(P, Q, R, [0,0.5,1,1.5,1.7579,2,2.5,3,4])
    ax2.clabel(c,inline=1,fontsize=10)


def f(x):
    return x**4+6*x**3+5*x**2-12*x-16

def aaa():
    X = np.linspace(1.4533, 1.4534, 100)
    print(X)
    for x in X:
        print(x, f(x))


if __name__ == '__main__':


    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()

    ax1 = fig.add_subplot(121, projection='3d')
    P,Q,R = draw_left_3d(ax1)
    ax2 = fig.add_subplot(122)
    draw_right_2d2(ax2, P, Q, R)

    plt.show()

