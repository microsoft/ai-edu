
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
import matplotlib.cm as cm
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
        X, Y = make_moons(n_samples=n_samples, noise=0.3, shuffle=False)
        Y[Y == 0] = -1
        samples = np.hstack((X,Y.reshape(-1,1)))
        np.savetxt(file_path, samples, fmt='%f, %f, %d', delimiter=',', header='x1, x2, y')
    return X, Y


# 绘图区基本设置
def set_ax(ax, scope):
    if (ax.name != '3d'):
        ax.grid()
        ax.axis('equal')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    if (scope is not None):
        ax.set_xlim(scope[0], scope[1])
        ax.set_ylim(scope[3], scope[4])

# 绘制平面样本点
def draw_2d_samples(ax, x, y, display_text=True):
    ax.scatter(x[y==1,0], x[y==1,1], marker='^', color='red')
    ax.scatter(x[y==-1,0], x[y==-1,1], marker='o', color='blue')
    if (display_text):
        for i in range(x.shape[0]):
            ax.text(x[i,0], x[i,1]+0.1, str(i), clip_on=True)

 

# 显示分类区域结果
def show_predication_result(ax, model, X, Y, scope):   
    set_ax(ax, scope)

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

    # 绘图
    cmap = ListedColormap(['yellow','lightgray'])
    plt.contourf(X1,X2, y_pred, cmap=cmap)
    # 绘制原始样本点用于比对
    if (X.shape[0]<=10):
        draw_2d_samples(ax, X, Y, display_text=True)
    else:
        draw_2d_samples(ax, X, Y, display_text=False)


# 高斯核函数 SVM
def rbf_svc(X, Y, C, gamma):
    model = SVC(C=C, gamma = gamma, kernel='rbf')
    model.fit(X,Y)

    #print("权重:",np.dot(model.dual_coef_, model.support_vectors_))
    print("支持向量个数:",model.n_support_)
    print("支持向量索引:",model.support_)
    print("支持向量:",np.round(model.support_vectors_,3))
    print("支持向量ay:",np.round(model.dual_coef_,3))
    print("偏移值:",model.intercept_)
    print("准确率:", model.score(X, Y))

    return model

def compare_3_gamma(X, Y, gamma):

    scope = [-1,1,20,-1,1,20]
    plt.axis('off')
    C = 1

    for i in range(3):
        gamma_str = str.format("gamma={0}", gamma[i])
        print(gamma_str)
        model = rbf_svc(X, Y, C, gamma[i])
        ax = plt.subplot(1,3,i+1)
        ax.set_title(gamma_str)
        show_predication_result(ax, model, X, Y, scope)
        ax.scatter(0,0, marker='x')      

        num = np.sum(model.n_support_)
        sum = 0
        for j in range(num):
            sum += model.dual_coef_[0,j] * np.exp(-gamma[i] * np.linalg.norm(model.support_vectors_[j])**2)
        distance = sum + model.intercept_
        print("手工计算结果:", distance)
        print("判定函数结果:", model.decision_function([[0,0]]))
        print("预测分类结果:", model.predict([[0,0]]))


    plt.show()


if __name__=="__main__":
    # 10 个样本
    file_name = "Data_12_moon_10.csv"
    X_raw, Y = load_data(file_name, 10)
    # 标准化
    ss = StandardScaler()
    X = ss.fit_transform(X_raw)
    # 比较分类区域
    gamma = [1,5,10]
    compare_3_gamma(X, Y, gamma)

