import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def TargetFunction(x):
    p1 = 0.4 * (x**2)
    p2 = 0.3 * x * np.sin(15 * x)
    p3 = 0.01 * np.cos(50 * x)
    y = p1 + p2 + p3 - 0.3
    return y

def CreateSampleData(num_train, num_test):
    # create train data
    x1 = np.random.random((num_train,1))
    y1 = TargetFunction(x1) + (np.random.random((num_train,1))-0.5)/10

    # create test data
    x2 = np.linspace(0,1,num_test).reshape(num_test,1)
    y2 = TargetFunction(x2)

    return x1, y1, x2, y2

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
    return A

def make_x(x1, m):
    row = x1.shape[0]
    Xm = np.zeros((row, m))
    for i in range(m):
        Xm[:,i:i+1] = np.power(x1, i+1).reshape(row,1)
    return Xm


def show_result(X,Y,A):
    # 样本点
    plt.scatter(X[:,0],Y)
    
    # 回归曲线
    row = 50
    col = A.shape[0]    
    x = np.linspace(0,1,row)
    X = make_x(x, col-1)
    ones = np.ones((row, 1))
    X = np.column_stack((ones, X))  
    Y = np.dot(X, A)
    plt.plot(x, Y, color='Red')
    
    plt.show()

def pred(A, X, Y):
    ones = np.ones((Y.shape[0], 1))
    Xm = make_x(X, A.shape[0]-1)
    X = np.column_stack((ones, Xm))  
    y_hat = np.dot(X, A)
    loss = np.sum((Y-y_hat)*(Y-y_hat))/Y.shape[0]/2
    print(str.format("loss={0:.6f}",loss))
    print(CalAccuracy(y_hat, Y))

def CalAccuracy(y_hat, y):
    assert(y_hat.shape == y.shape)
    m = y_hat.shape[0]
    var = np.var(y)
    mse = np.sum((y_hat-y)**2)/m
    r2 = 1 - mse / var
    return r2


if __name__ == '__main__':
    X1, Y1, X2, Y2 = CreateSampleData(100, 50)
    X = make_x(X1, 30)
    A = normal_equation(X,Y1)
    print(A)
    show_result(X,Y1,A)
    pred(A, X2, Y2)