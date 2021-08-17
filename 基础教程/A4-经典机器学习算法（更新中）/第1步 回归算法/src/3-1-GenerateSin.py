import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def normal_equation(X,Y):
    num_example = X.shape[0]
    # 在原始的X矩阵最左侧加一列1
    ones = np.ones((num_example,1))
    x = np.column_stack((ones, X))    
    # X^T * X
    p = np.dot(x.T, x)
    # (X^T * X)^{-1}
    q = np.linalg.inv(p)
    # (X^T * X)^{-1} * X^T
    r = np.dot(q, x.T)
    # (X^T * X)^{-1} * X^T * Y
    A = np.dot(r, Y)
    # 按顺序
    b = A[0,0]
    a1 = A[1,0]
    a2 = A[2,0]
    return a1, a2, b

def show_result(X1,Y,a1,a2,b):
    plt.scatter(X1,Y)
    x = np.linspace(0,5,50)
    y = a1 * x + a2 * x * x + b
    plt.plot(x,y)
    plt.show()

if __name__ == '__main__':
    X1 = np.array([1, 2, 3, 4]).reshape(4,1)
    Y = np.array([1, 2 ,2.5 ,2]).reshape(4,1)
    # model y = a1*x + a2 * x^2 + b
    X2 = (X1*X1).reshape(4,1)
    X = np.column_stack((X1,X2))
    a1, a2, b = normal_equation(X,Y)
    print(a1, a2, b)
    show_result(X1,Y,a1,a2,b)


