import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

def load_file(file_name):
    file_path = generate_file_path(file_name)
    file = Path(file_path)
    if file.exists():
        samples = np.loadtxt(file, delimiter=',')
        return samples
    else:
        return None

def generate_file_path(file_name):
    curr_path = sys.argv[0]
    curr_dir = os.path.dirname(curr_path)
    file_path = os.path.join(curr_dir, file_name)
    return file_path

# 在原始的X矩阵最左侧加一列1
def add_ones_at_left(X0):
    num_example = X0.shape[0]
    ones = np.ones((num_example,1))
    X = np.column_stack((ones, X0))    
    return X

def normal_equation(X0,Y):
    X = add_ones_at_left(X0)
    # X^T * X
    p = np.dot(X.T, X)
    # (X^T * X)^{-1}
    q = np.linalg.inv(p)
    # (X^T * X)^{-1} * X^T
    r = np.dot(q, X.T)
    # (X^T * X)^{-1} * X^T * Y
    A = np.dot(r, Y)
    # 按顺序
    b = A[0,0]
    a1 = A[1,0]
    a2 = A[2,0]
    return a1, a2, b

def show_result(X, Y, a1, a2, b): 
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    
    # 准备拟合平面数据
    axis_x = np.linspace(0, 25, 50)
    axis_y = np.linspace(40, 120, 100)
    P,Q = np.meshgrid(axis_x, axis_y)
    R = a1 * P + a2 * Q + b
    
    fig = plt.figure()
    plt.title(u"给定位置和面积的房价预测")
    plt.axis('off')
    # 绘制左视图
    ax = fig.add_subplot(121,projection='3d')
    ax.set_xlabel(u"距离")
    ax.set_ylabel(u"面积")
    ax.set_zlabel(u"价格")
    # 绘制全部分样本点
    ax.scatter(X[:,0], X[:,1], Y)
    ax.plot_surface(P, Q, R, alpha=0.5, color='Red')

    ax = fig.add_subplot(122,projection='3d')
    ax.set_xlabel(u"距离")
    ax.set_ylabel(u"面积")
    ax.set_zlabel(u"价格")
    # 绘制一部分样本点
    ax.scatter(X[:,0], X[:,1], Y)
    ax.plot_surface(P, Q, R, alpha=0.5, color='Red')
    plt.show()

if __name__ == '__main__':
    file_name = "2-0-data.csv"
    samples = load_file(file_name)
    if (samples is not None):
        X = samples[:,0:2]
        Y = samples[:,-1].reshape(-1,1)
        a1, a2, b = normal_equation(X,Y)
        print(str.format("a1={0:.4f}, a2={1:.4f}, b={2:.4f}", a1, a2, b))
        show_result(X, Y, a1, a2, b)
    else:
        print("cannot find data file:" + file_name)
