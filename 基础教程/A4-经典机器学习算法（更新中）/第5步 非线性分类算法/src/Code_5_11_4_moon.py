
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from numpy.linalg.linalg import norm
from sklearn.datasets import *
from sklearn.svm import *
from sklearn.pipeline import *
from sklearn.preprocessing import *
import sys
import os
from pathlib import Path
import matplotlib as mpl
from Code_5_11_3_xor import *

def generate_file_path(file_name):
    curr_path = sys.argv[0]
    curr_dir = os.path.dirname(curr_path)
    file_path = os.path.join(curr_dir, file_name)
    return file_path

def load_data(file_name, n_samples):
    file_path = generate_file_path(file_name)
    file = Path(file_path)
    if file.exists():
        samples = np.loadtxt(file, delimiter=',')
        X = samples[:, 0:2]
        Y = samples[:, 2]
    else:
        X, Y = make_moons(n_samples=n_samples, noise=0.1, shuffle=False)
        Y[Y == 0] = -1
        samples = np.hstack((X,Y.reshape(-1,1)))
        np.savetxt(file_path, samples, fmt='%f, %f, %d', delimiter=',', header='x1, x2, y')
    return X, Y

def set_ax(ax):
    ax.grid()
    ax.axis('equal')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

def show_samples(X_10, Y_10, X_100, Y_100):
    # 基本绘图设置
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.axis('off')
    plt.title(u"月亮数据集的100个样本（左）和10个样本（右）")

    ax1 = fig.add_subplot(121)
    set_ax(ax1)
    draw_2d(ax1, X_100, Y_100)

    ax2 = fig.add_subplot(122)
    set_ax(ax2)
    draw_2d(ax2, X_10, Y_10)

    plt.show()

def draw_2d(ax, x, y):
    ax.scatter(x[y==1,0], x[y==1,1], marker='^')
    ax.scatter(x[y==-1,0], x[y==-1,1], marker='o')


if __name__=="__main__":

    file_name = "Data_moon_10.csv"
    X_10, Y_10 = load_data(file_name, 10)

    file_name = "Data_moon_100.csv"
    X_100, Y_100 = load_data(file_name, 100)

    #show_samples(X_10, Y_10, X_100, Y_100)

    ss = StandardScaler()
    X = ss.fit_transform(X_10)
    print("X 标准化后的值：")
    print(X)

    gamma = 2
    X_new = K_matrix(X, X, gamma)
    print("映射结果：")
    print(np.round(X_new,3))

    # 尝试用线性 SVM 做分类    
    C = 3
    model = linear_svc(X_new, Y_10, C)
    # 显示分类预测结果
    X1, X2, y_pred = prediction(model, gamma, X, [-2,2,100,-2,2,100])
    show_result(X1, X2, y_pred, X, Y_10)
