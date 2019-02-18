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
        return X,Y
    else:
        return None,None

if __name__ == '__main__':
    X,Y = ReadData()

    m = X.shape[0]
    x_sum = sum(X)
    x_mean = x_sum/m
    x_square = sum(X*X)
    x_square_mean = x_sum * x_sum / m
    xy = sum(Y*(X-x_mean))
    w = xy / (x_square - x_square_mean)
    b = sum(Y-w*X) / m
    print("w=%f, b=%f" % (w,b))

