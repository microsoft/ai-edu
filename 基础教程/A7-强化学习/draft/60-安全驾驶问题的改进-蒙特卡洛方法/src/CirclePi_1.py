import random
from turtle import color
import matplotlib.pyplot as plt
import numpy as np

# 画圆
def draw_circle(ax,r,x,y):
    # 点的横坐标为a
    a = np.arange(x-r,x+r,0.000001)
    # 点的纵坐标为b
    b = np.sqrt(np.power(r,2)-np.power((a-x),2))+y
    ax.plot(a,b,color='g',linestyle='-')
    ax.plot(a,-b,color='g',linestyle='-')

# 随机投点
def put_points(ax, num_total_points):
    ax.axis('equal')
    data = np.random.uniform(-1, 1, size=(num_total_points, 2))     # 在正方形内生成随机点
    r = np.sqrt(np.power(data[:,0], 2) + np.power(data[:,1], 2))    # 计算每个点到中心的距离
    num_in_circle = 0   # 统计在圆内的点数
    for i, point in enumerate(data):    # 绘图
        if (r[i] < 1):
            num_in_circle += 1     # 计数
            ax.plot(point[0], point[1], 'o', markersize=1, c='r')
        else:
            ax.plot(point[0], point[1], 'o', markersize=1, c='b')
    # 计算 pi 值
    title = str.format("n={0},$\pi$={1}",num_total_points, num_in_circle/num_total_points*4)
    ax.set_title(title)
    draw_circle(ax, 1, 0, 0)
    ax.grid()

if __name__=="__main__":
    fig = plt.figure()

    ax = fig.add_subplot(141)
    put_points(ax, 100)
    ax = fig.add_subplot(142)
    put_points(ax, 200)
    ax = fig.add_subplot(143)
    put_points(ax, 500)
    ax = fig.add_subplot(144)
    put_points(ax, 1000)

    plt.show()
