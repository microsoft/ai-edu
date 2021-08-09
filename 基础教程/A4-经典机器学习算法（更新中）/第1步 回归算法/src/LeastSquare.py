import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.pyplot as mpl

# 根据公式1.8
def calculate_a(X,Y):
    n = X.shape[0]
    numerator = n * np.sum(X*Y) - np.sum(X) * np.sum(Y)
    denominator = n * np.sum(X*X) - np.sum(X) * np.sum(X)
    a = numerator / denominator
    return a

# 根据公式1.7
def calculate_b(a,X,Y):
    n = X.shape[0]
    b = (np.sum(Y - a * X))/n
    return b

def generate_file_path(file_name):
    curr_path = sys.argv[0]
    curr_dir = os.path.dirname(curr_path)
    file_path = os.path.join(curr_dir, file_name)
    return file_path

def least_square(X,Y):
    a_hat = calculate_a(X,Y)
    b_hat = calculate_b(a_hat,X,Y)
    print(str.format("a_hat={0}, b_hat={1}",a_hat,b_hat))
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
        a_hat, b_hat = least_square(X,Y)
        show_result(X,Y,a_hat,b_hat)
    else:
        print("cannot find file " + file_name)