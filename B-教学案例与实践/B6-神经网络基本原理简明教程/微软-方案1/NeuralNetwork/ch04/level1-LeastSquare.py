# 用最小二乘法得到解析解LSE

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

x_data_name = "TemperatureControlXData.dat"
y_data_name = "TemperatureControlYData.dat"

def ReadData():
    Xfile = Path(x_data_name)
    Yfile = Path(y_data_name)
    if Xfile.exists() & Yfile.exists():
        X = np.load(Xfile)
        Y = np.load(Yfile)
        return X, Y
    else:
        return None,None

def method1(X,Y,m):
    x_mean = np.mean(X)
    a = sum(Y*(X-x_mean))
    w = a / (sum(X*X) - sum(X)*sum(X)/m)
    return w

def method2(X,Y,m):
    x_mean = sum(X)/m
    y_mean = sum(Y)/m
    w = sum(X*(Y-y_mean)) / (sum(X*X) - x_mean*sum(X))
    return w

def method3(X,Y,m):
    m = X.shape[0]
    a = m*sum(X*Y) - sum(X)*sum(Y)
    b = m*sum(X*X) - sum(X)*sum(X)
    w = a/b
    return w

if __name__ == '__main__':
    X,Y = ReadData()
    m = X.shape[0]
    w1 = method1(X,Y,m)
    b1 = sum(Y-w1*X) / m

    w2 = method2(X,Y,m)
    b2 = sum(Y-w2*X) / m

    w3 = method3(X,Y,m)
    b3 = sum(Y-w3*X) / m

    print("w1=%f, b1=%f" % (w1,b1))
    print("w2=%f, b2=%f" % (w2,b2))
    print("w3=%f, b3=%f" % (w3,b3))

