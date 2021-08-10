import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.pyplot as mpl


def generate_file_path(file_name):
    curr_path = sys.argv[0]
    curr_dir = os.path.dirname(curr_path)
    file_path = os.path.join(curr_dir, file_name)
    return file_path

def least_square_2(X,Y):
    # 公式1.17
    numerator = np.cov(X, Y, rowvar=False)[0,1]
    denominator = np.var(X)
    a_hat = numerator / denominator
    # 公式1.18
    b_hat = np.mean(Y) - a_hat * np.mean(X)
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

def load_file(file_name):
    file_path = generate_file_path(file_name)
    file = Path(file_path)
    if file.exists():
        samples = np.loadtxt(file, delimiter=',')
        return samples
    else:
        return None

if __name__ == '__main__':
    file_name = "01-linear.csv"
    samples = load_file(file_name)
    if samples is not None:
        X = samples[:, 0]
        Y = samples[:, 1]
        a_hat, b_hat = least_square_2(X,Y)
        show_result(X,Y,a_hat,b_hat)
    else:
        print("cannot find file " + file_name)