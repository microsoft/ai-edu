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
def show_result(X, Y, A3, A4):
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.title(u"回归效果比较")
    plt.axis("off")

    # 绘制左视图
    ax = fig.add_subplot(121)
    ax.set_xlabel(u"三次多项式回归效果")
    ax.grid()
    # 样本点
    ax.scatter(X,Y)
    # 回归直线
    count = 50
    # 按顺序从0到6给出50个连续样本点做回归预测
    x = np.linspace(1,7,count).reshape(-1,1)
    y_hat = prediction(x, A3)
    plt.plot(x, y_hat, color='Red')


    # 绘制右视图
    ax = fig.add_subplot(122)
    ax.set_xlabel(u"四次多项式回归效果")
    ax.grid()
    # 样本点
    ax.scatter(X,Y)
    # 回归直线
    count = 50
    # 按顺序从0到6给出50个连续样本点做回归预测
    x = np.linspace(1,7,count).reshape(-1,1)
    y_hat = prediction(x, A4)
    plt.plot(x, y_hat, color='Red')

    plt.show()

def mse(y, y_hat):
    loss = np.sum((y - y_hat)*(y - y_hat))
    return loss

if __name__ == '__main__':
    # 图A
    X1 = np.array([1, 1.5, 2, 3, 4, 5, 6, 6.5]).reshape(-1,1)
    Y = np.array([1, 2, 3, 3.5, 3, 2.5, 3, 4]).reshape(-1,1)

    # 多项式回归
    Xm = make_Xm(X1, 3)
    A3 = normal_equation(Xm,Y)
    print("# 三次多项式回归结果")
    print(str.format("a1={0:.4f}, a2={1:.4f}, b={2:.4f}", A3[1,0], A3[2,0], A3[0,0]))
    
    Xm = make_Xm(X1, 4)
    A4 = normal_equation(Xm,Y)
    print("# 四次多项式回归结果")
    print(str.format("a1={0:.4f}, a2={1:.4f}, a3={2:.4f}, b={3:.4f}", A4[1,0], A4[2,0], A4[3,0], A4[0,0]))
    show_result(X1, Y, A3, A4)
    
    Y_hat3 = prediction(X1, A3)
    J3 = mse(Y, Y_hat3)
    Y_hat4 = prediction(X1, A4)
    J4 = mse(Y, Y_hat4)
    print("# 损失函数值")
    print(str.format("J3={0:.4f}, J4={1:.4f}", J3, J4))
