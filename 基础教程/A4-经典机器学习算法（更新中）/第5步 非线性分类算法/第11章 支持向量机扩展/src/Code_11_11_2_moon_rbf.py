
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
from matplotlib.colors import ListedColormap

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

# 绘图区基本设置
def set_ax(ax, scope):
    ax.axis('equal')
    ax.grid()
    ax.xlabel("x1")
    ax.ylabel("x2")
    if (scope is not None):
        ax.xlim(scope[0], scope[1])
        ax.ylim(scope[3], scope[4])

# 绘制平面样本点
def draw_2d_samples(ax, x, y, display_text=True):
    ax.scatter(x[y==1,0], x[y==1,1], marker='^', color='red')
    ax.scatter(x[y==-1,0], x[y==-1,1], marker='o', color='blue')
    if (display_text):
        for i in range(x.shape[0]):
            ax.text(x[i,0], x[i,1]+0.1, str(i))

 

# 显示高斯核函数，即所有样本点之间的内积
def show_result_2(X1, X2, y_pred, X, Y):
    # 基本绘图设置
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.title(title)

    set_ax(plt, scope)
    # 绘图
    cmap = ListedColormap(['yellow','lightgray'])
    plt.contourf(X1,X2, y_pred, cmap=cmap)
    # 绘制原始样本点用于比对
    draw_2d_samples(plt, X, Y)

    plt.show()


def rbf_svc(X, Y, C, gamma):
    model = SVC(C=C, gamma = gamma, kernel='rbf')
    model.fit(X,Y)

    #print("权重:",model.coef_)
    print("支持向量个数:",model.n_support_)
    print("支持向量索引:",model.support_)
    print("支持向量:",np.round(model.support_vectors_,3))
    print("支持向量ay:",model.dual_coef_)
    print("准确率:", model.score(X, Y))

    return model

# 生成测试数据，形成一个点阵来模拟平面
def prediction(model, scope):
    # 生成测试数据，形成一个点阵来模拟平面
    x1 = np.linspace(scope[0], scope[1], scope[2])
    x2 = np.linspace(scope[3], scope[4], scope[5])
    X1,X2 = np.meshgrid(x1,x2)
    # 从行列变形为序列数据
    X12 = np.c_[X1.ravel(), X2.ravel()]
    # 做预测
    pred = model.predict(X12)
    # 从序列数据变形行列形式
    y_pred = pred.reshape(X1.shape)

    return X1, X2, y_pred


if __name__=="__main__":

    file_name = "Data_moon_10.csv"
    X_raw, Y = load_data(file_name, 10)

    ss = StandardScaler()
    X = ss.fit_transform(X_raw)
    print("X 标准化后的值：")
    print(X)

    # 用rbf SVM 做分类    
    gamma = 2
    C = 1
    model = rbf_svc(X, Y, C, gamma)
    # 显示分类预测结果
    scope = [-3,3,100,-3,3,100]
    X1, X2, y_pred = prediction(model, scope)
    
    title = u"所有样本（含权重）的高斯核函数示意图"
    show_result_2(X1, X2, y_pred, X, Y)
