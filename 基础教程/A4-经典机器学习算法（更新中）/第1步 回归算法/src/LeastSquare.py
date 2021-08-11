import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl


def generate_file_path(file_name):
    curr_path = sys.argv[0]
    curr_dir = os.path.dirname(curr_path)
    file_path = os.path.join(curr_dir, file_name)
    return file_path

# 公式 1.2.5
def least_square_1(X,Y):
    n = X.shape[0]
    # a_hat
    numerator = n * np.sum(X*Y) - np.sum(X) * np.sum(Y)
    denominator = n * np.sum(X*X) - np.sum(X) * np.sum(X)
    a_hat = numerator / denominator
    # b_hat
    b_hat = (np.sum(Y - a_hat * X))/n
    return a_hat, b_hat

# 公式 1.4.2
def least_square_2(X,Y):
    n = X.shape[0]
    # a_hat
    numerator = np.cov(X, Y, rowvar=False, bias=False)[0,1]
    denominator = np.var(X)
    a_hat = numerator / denominator
    # b_hat
    b_hat = (np.sum(Y - a_hat * X))/n
    return a_hat, b_hat

# 公式1.4.3, 1.4.4
def least_square_3(X,Y):
    n = X.shape[0]
    # a_hat 公式1.4.3
    numerator = np.sum((X-np.mean(X))*(Y-np.mean(Y)))
    denominator = np.sum((X-np.mean(X))*(X-np.mean(X)))
    a_hat = numerator / denominator
    # b_hat 公式1.4.4
    b_hat = np.mean(Y) - a_hat * np.mean(X)
    return a_hat, b_hat

def show_result(X,Y,a_hat,b_hat):
    # 用来正常显示中文标签
    mpl.rcParams['font.sans-serif'] = ['DengXian']  
    mpl.rcParams['axes.unicode_minus']=False
    plt.scatter(X,Y,s=10)
    plt.title(u"机房空调功率预测")
    plt.xlabel(u"服务器数量(千台)")
    plt.ylabel(u"空调功率(千瓦)")
    x = np.linspace(0,1)
    y = a_hat * x + b_hat
    plt.plot(x,y)
    plt.show()


if __name__ == '__main__':
    file_name = "01-linear.csv"
    file_path = generate_file_path(file_name)
    file = Path(file_path)
    if file.exists():
        samples = np.loadtxt(file, delimiter=',')
        X = samples[:, 0]
        Y = samples[:, 1]
        a_hat, b_hat = least_square_1(X,Y)
        print(str.format("a_hat={0:.4f}, b_hat={1:.4f}",a_hat,b_hat))
        a_hat, b_hat = least_square_2(X,Y)
        print(str.format("a_hat={0:.4f}, b_hat={1:.4f}",a_hat,b_hat))
        a_hat, b_hat = least_square_3(X,Y)
        print(str.format("a_hat={0:.4f}, b_hat={1:.4f}",a_hat,b_hat))

        show_result(X,Y,a_hat,b_hat)
    else:
        print("cannot find file " + file_name)