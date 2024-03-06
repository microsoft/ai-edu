import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

# 生成当前目录的全文件名
def generate_file_path(file_name):
    curr_path = sys.argv[0]
    curr_dir = os.path.dirname(curr_path)
    file_path = os.path.join(curr_dir, file_name)
    return file_path

# 加载csv样本文件
def load_file(file_name):
    file_path = generate_file_path(file_name)
    file = Path(file_path)
    if file.exists():
        samples = np.loadtxt(file, delimiter=',')
        return samples
    else:
        return None

# 公式 1.3.8
def least_square_1(X,Y):
    n = X.shape[0]
    # a_hat
    numerator = n * np.sum(X*Y) - np.sum(X) * np.sum(Y)
    denominator = n * np.sum(X*X) - np.sum(X) * np.sum(X)
    a_hat = numerator / denominator
    # b_hat
    b_hat = (np.sum(Y - a_hat * X))/n
    return a_hat, b_hat

# 可视化
def show_result(X,Y,a_hat,b_hat):
    # 用来正常显示中文标签
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    plt.scatter(X,Y,s=10)
    plt.title(u"机房空调功率预测")
    plt.xlabel(u"服务器数量(千台)")
    plt.ylabel(u"空调功率(千瓦)")
    x = np.linspace(0,1)
    y = a_hat * x + b_hat
    plt.plot(x,y)
    plt.show()

# 计算均方差
def calculate_mse(Y, Y_hat):
    loss = np.sum((Y-Y_hat)*(Y-Y_hat))/Y.shape[0]
    return loss

# 根据参数 a 和 b 计算模型回归值 Y_hat，然后与 Y 做均方差
def calculate_J(a,b,X,Y):
    Y_hat = a * X + b
    J = calculate_mse(Y, Y_hat)
    return J


if __name__ == '__main__':
    file_name = "1-0-data.csv"
    samples = load_file(file_name)
    if samples is not None:
        X = samples[:, 0].reshape(200,1)
        Y = samples[:, 1].reshape(200,1)
        a_hat, b_hat = least_square_1(X,Y)
        print(str.format("a_hat={0:.4f}, b_hat={1:.4f}",a_hat,b_hat))

        show_result(X,Y,a_hat,b_hat)

        # 比较估计值和原始的均方差的大小
        J1 = calculate_J(a_hat, b_hat, X, Y)
        J2 = calculate_J(0.5, 1, X, Y)
        print(str.format("J1={0:.6f}, J2={1:.6f}", J1, J2))

    else:
        print("cannot find file " + file_name)
