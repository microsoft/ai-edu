import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.core.fromnumeric import size

# 在原始的X矩阵最左侧加一列1
def add_ones_at_left(X0):
    num_example = X0.shape[0]
    ones = np.ones((num_example,1))
    X = np.column_stack((ones, X0))    
    return X

# 解正规方程
# X0 - 还没有在左侧加 1 的原始样本值
# Y - X0 对应的标签值
def normal_equation(X0,Y):
    # 在原始的X矩阵最左侧加一列1
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
    return A

# 给原始的一维 X 增加到 m 维多项式
def make_Xm(X1, m):
    count = X1.shape[0]
    Xm = np.zeros((count, m))
    for i in range(m):
        Xm[:,i:i+1] = np.power(X1, i+1)
    return Xm

def prediction(X1,A):
    # 增加多项式
    Xm = make_Xm(X1, A.shape[0]-1)
    # 左侧增加一列 1
    ones_X = add_ones_at_left(Xm)
    Y_hat = np.dot(ones_X, A)
    return Y_hat

# X，Y - 原始样本点
# A1 - 线性回归参数值
# A - 多项式回归参数值
def show_result(Xm, Y, A):
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.title(u"多项式回归原理解释")
    plt.axis('off')

    # 左子图显示样本点
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(Xm[:,0], Xm[:,1], Y)
    ax.set_xlabel(u"原始样本点特征")
    ax.set_ylabel(u"平方值特征")
 
    # 准备拟合平面数据
    axis_x = np.linspace(0, 7, 50)
    axis_y = np.linspace(0, 35, 50)
    P,Q = np.meshgrid(axis_x, axis_y)
    R = A[1,0] * P + A[2,0] * Q + A[0,0]
    
    # 中图显示平面拟合效果
    ax = fig.add_subplot(132, projection='3d')
    ax.set_xlabel(u"原始样本点特征")
    ax.set_ylabel(u"平方值特征")

    # 样本点
    ax.scatter(Xm[:,0], Xm[:,1], Y)
    ax.plot_surface(P, Q, R, alpha=0.5, color='Red')
    ax.set_zlim((1,4))

    # 中图显示平面拟合效果
    ax = fig.add_subplot(133, projection='3d')
    ax.set_xlabel(u"原始样本点特征")
    ax.set_ylabel(u"平方值特征")

    # 样本点
    ax.scatter(Xm[:,0], Xm[:,1], Y)
    ax.plot_surface(P, Q, R, alpha=0.5, color='Red')
    ax.set_zlim((1,4))


    plt.show()


if __name__ == '__main__':
    # 图B
    X1 = np.array([1, 2, 3, 4, 5, 6]).reshape(-1,1)
    Y = np.array([4, 3, 2, 1.5, 2, 3]).reshape(-1,1)

    # 多项式回归
    Xm = make_Xm(X1, 2)
    A = normal_equation(Xm,Y)
    show_result(Xm, Y, A)

