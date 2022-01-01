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
    print("准确率:", score)

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
def show_predication_result(ax, model, X, Y, scope, style='binary'):
    # 生成测试数据，形成一个点阵来模拟平面
    x1 = np.linspace(scope[0], scope[1], scope[2])
    x2 = np.linspace(scope[3], scope[4], scope[5])
    X1,X2 = np.meshgrid(x1,x2)
    # 从行列变形为序列数据
    X12 = np.c_[X1.ravel(), X2.ravel()]
    # 做预测
    if (style == 'binary'):
        pred = model.predict(X12)   # +1/-1
    else:
        pred = model.decision_function(X12)     # distance, float number
    # 从序列数据变形行列形式
    y_pred = pred.reshape(X1.shape)

    # 绘图
    if (style == 'binary'):
        cmap = ListedColormap(['yellow','lightgray'])
        plt.contourf(X1,X2, y_pred, cmap=cmap)
    else:
        plt.contourf(X1,X2, y_pred)
    # 绘制原始样本点用于比对
    if (X.shape[0]<=10):
        draw_2d_samples(ax, X, Y, display_text=True)
    else:
        draw_2d_samples(ax, X, Y, display_text=False)

def classification(X_raw, Y, hasCoef0):
    ss = StandardScaler()
    X = ss.fit_transform(X_raw)

    fig = plt.figure()
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False

    scope = [-2.5,2.5,100,-2.5,2.5,100]    

    degree = [2,3,4,5,6,7]
    if (hasCoef0):  # 是否有 r 值
        coef0 = [1,1,1,1,1,1]
    else:
        coef0 = [0,0,0,0,0,0]
    for i in range(2):
        for j in range(3):
            idx = i * 3 + j
            d = degree[idx]
            r = coef0[idx]
            model, score = poly_svc(X, Y, d, r)
            ax = plt.subplot(2,3,idx+1)
            set_ax(ax, scope)
            title = str.format("degree={0},coef0={1}, 准确率={2:.2f}", d, r, score)
            ax.set_title(title)
            show_predication_result(ax, model, X, Y, scope, style='detail')

    plt.show()


if __name__=="__main__":

    X_raw = np.array([[-1.5,0], [-1,0], [-0.5,0], [0,0], [0.5,0], [1,0], [1.5,0]])
    Y = np.array([-1,-1,1,1,1,-1,-1])
    classification(X_raw, Y, False)

    X_raw = np.array([[-1.5,0], [-1,0], [-0.5,0], [0,0], [0.5,0], [1,0], [1.5,0]])
    Y = np.array([-1,-1,1,1,1,-1,-1])
    classification(X_raw, Y, True)
