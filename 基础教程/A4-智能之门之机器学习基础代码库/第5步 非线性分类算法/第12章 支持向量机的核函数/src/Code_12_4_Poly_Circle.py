import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.svm import SVC
from sklearn.preprocessing import *
import sys
import os
from pathlib import Path
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import time

def generate_file_path(file_name):
    curr_path = sys.argv[0]
    curr_dir = os.path.dirname(curr_path)
    file_path = os.path.join(curr_dir, file_name)
    return file_path

def load_data(file_name):
    file_path = generate_file_path(file_name)
    file = Path(file_path)
    if file.exists():
        samples = np.loadtxt(file, delimiter=',')
        X = samples[:, 0:2]
        Y = samples[:, 2]
    else:
        X, Y = make_circles(n_samples=100, factor=0.5, noise=0.1)
        Y[Y == 0] = -1
        samples = np.hstack((X,Y.reshape(-1,1)))
        np.savetxt(file_path, samples, fmt='%f, %f, %d', delimiter=',', header='x1, x2, y')
    return X, Y


# 绘制平面样本点
def draw_2d_samples(ax, x, y, display_text=True):
    ax.scatter(x[y==1,0], x[y==1,1], marker='^', color='red')
    ax.scatter(x[y==-1,0], x[y==-1,1], marker='o', color='blue')
    if (display_text):
        for i in range(x.shape[0]):
            ax.text(x[i,0], x[i,1]+0.1, str(i))


# 多项式核函数 SVM
def poly_svc(X, Y, d, r):
    model = SVC(kernel='poly', degree=d, coef0=r)
    model.fit(X,Y)

    #print("权重:",np.dot(model.dual_coef_, model.support_vectors_))
    print("支持向量个数:",model.n_support_)
    #print("支持向量索引:",model.support_)
    #print("支持向量:",np.round(model.support_vectors_,3))
    #print("支持向量ay:",np.round(model.dual_coef_,3))
    score = model.score(X, Y)
    #print("准确率:", score)

    return model, score

# 绘图区基本设置
def set_ax(ax, scope):
    ax.axis('equal')
    ax.grid()
    #ax.set_xlabel("x1")
    #ax.set_ylabel("x2")
    if (scope is not None):
        ax.set_xlim(scope[0], scope[1])
        ax.set_ylim(scope[3], scope[4])

# 显示分类区域结果
# style = 
# 'binary': 绘制双色区域分类图
# 'detail': 绘制渐变色区域分类图
# 'contour': 绘制双色区域分类图 + 分类间隔
def show_predication_result(ax, model, X, Y, scope, style='binary'):
    # 生成测试数据，形成一个点阵来模拟平面
    x1 = np.linspace(scope[0], scope[1], scope[2])
    x2 = np.linspace(scope[3], scope[4], scope[5])
    X1,X2 = np.meshgrid(x1,x2)
    # 从行列变形为序列数据
    X12 = np.c_[X1.ravel(), X2.ravel()]

    cmap = ListedColormap(['yellow','lightgray'])
    # 绘图
    if (style == 'binary'):
        pred = model.predict(X12)   # +1/-1
        y_pred = pred.reshape(X1.shape)
        plt.contourf(X1,X2, y_pred, cmap=cmap)
    elif (style=='detail'):
        pred = model.decision_function(X12)     # distance, float number
        y_pred = pred.reshape(X1.shape)
        plt.contourf(X1,X2, y_pred)
    else: # contour
        pred = model.predict(X12)   # +1/-1
        y_pred = pred.reshape(X1.shape)
        plt.contourf(X1,X2, y_pred, cmap=cmap)

        pred = model.decision_function(X12)     # distance, float number
        y_pred = pred.reshape(X1.shape)
        plt.contour(X1,X2, y_pred, colors=['red', 'black', 'blue'], linestyles=['--','-','--'], levels=[-1,0,1])
    # 绘制原始样本点用于比对
    if (X.shape[0]<=10):
        draw_2d_samples(ax, X, Y, display_text=True)
    else:
        draw_2d_samples(ax, X, Y, display_text=False)

def classification(X_raw, Y, degree, coef0):
    ss = StandardScaler()
    X = ss.fit_transform(X_raw)

    fig = plt.figure()
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False

    scope = [-2.5,2.5,100,-2.5,2.5,100]    

    for i in range(2):
        for j in range(4):
            idx = i * 4 + j
            d = degree[idx]
            r = coef0[idx]
            model, score = poly_svc(X, Y, d, r)
            ax = plt.subplot(2,4,idx+1)
            set_ax(ax, scope)
            title = str.format("degree={0},coef0={1}, 准确率={2}", d, r, score)            
            print(title)
            ax.set_title(title)
            show_predication_result(ax, model, X, Y, scope, style='contour')

    plt.show()


if __name__=="__main__":

    X_raw, Y = load_data("Data_12_circle_100.csv")
    degree = [2,3,4,5,2,3,4,5]
    coef0 = [0,0,0,0,1,1,1,1]
    classification(X_raw, Y, degree, coef0)



 