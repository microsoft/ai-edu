import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.core.fromnumeric import size


def normal_equation(X, Y, r):
    num_example = X.shape[0]
    # 在原始的X矩阵最左侧加一列1
    ones = np.ones((num_example,1))
    x = np.column_stack((ones, X))    
    # X^T * X
    p = np.dot(x.T, x)
    # (X^T * X)^{-1}
    #I = np.eye(p.shape[0]) * 1e-6
    #p = p + I

    I = np.eye(p.shape[0])
    p = p + r * I

    q = np.linalg.inv(p)
    # (X^T * X)^{-1} * X^T
    r = np.dot(q, x.T)
    # (X^T * X)^{-1} * X^T * Y
    A = np.dot(r, Y)
    # 按顺序
    return A

def make_x(X, m):
    row = X.shape[0]
    Xm = np.zeros((row, m))
    for i in range(row):
        Xm[:,i:i+1] = np.power(X, i+1)
    return Xm

def show_result(X,Y,A):
    # 样本点
    plt.scatter(X[:,0],Y)
    # 回归曲线
    row = 50
    col = A.shape[0]
    X = np.zeros((row,col))
    x = np.linspace(5,19,row)
    for i in range(col):
        X[:,i] = np.power(x, i)
    Y = np.dot(X, A)
    plt.plot(x, Y)
    plt.show()

if __name__ == '__main__':
    X1 = np.array([6, 8, 10, 14, 18]).reshape(-1,1)
    Y = np.array([7, 9, 13, 17.5, 18]).reshape(-1,1)
    #Y = np.array([2, 1, 2 ,2.5 ,2, 1]).reshape(-1,1)
    X = make_x(X1, 8)
    r = 0.1 # 正则项参数
    A = normal_equation(X, Y, r)
    print(A)
    show_result(X,Y,A)


