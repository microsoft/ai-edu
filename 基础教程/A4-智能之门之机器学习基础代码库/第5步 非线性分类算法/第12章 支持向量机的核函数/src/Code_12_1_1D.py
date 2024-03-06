
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import *
import matplotlib as mpl

# 绘图区基本设置
def set_ax(ax, scope):
    ax.axis('equal')
    ax.grid()
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    if (scope is not None):
        ax.set_xlim(scope[0][0], scope[0][1])
        ax.set_ylim(scope[1][0], scope[1][1])

# 线性SVM分类器
def linear_svc(X,Y):
    model = SVC(C=3, kernel='linear')
    model.fit(X,Y)

    #print("权重:",model.coef_)
    print("支持向量个数:",model.n_support_)
    print("支持向量索引:",model.support_)
    print("支持向量:",np.round(model.support_vectors_,3))
    print("支持向量ay:",model.dual_coef_)
    print("准确率:", model.score(X, Y))

    return model

# 展示分类结果
def show_result(model, X_raw, X, Y):

    fig = plt.figure()
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False

    # 映射后的线性分类
    ax1 = fig.add_subplot(121)
    ax1.set_title(u"映射后的线性分类")
    scope = ((-2,2,100),(-1,3,100))
    set_ax(ax1, scope)
    x1 = np.linspace(*scope[0])
    x2 = np.linspace(*scope[1])
    X1,X2 = np.meshgrid(x1,x2)
    X12 = np.c_[X1.ravel(), X2.ravel()]
    pred = model.predict(X12)
    y_pred = pred.reshape(X1.shape)
    cmap = ListedColormap(['yellow','lightgray'])
    ax1.contourf(X1,X2, y_pred, cmap=cmap)
    # 升维后的样本数据
    show_samples(ax1, X, Y)

    # 压缩回到原始坐标系的分类界线
    ax2 = fig.add_subplot(122)
    ax2.set_title(u"压缩回到原始坐标系的分类界线")
    scope = ((-2,2,100),(-1,1,100))
    set_ax(ax2, scope)
    x1 = np.linspace(*scope[0])
    x2 = np.linspace(*scope[1])
    X1,X2 = np.meshgrid(x1,x2)
    X12 = np.c_[X1.ravel(), X2.ravel()]
    # 把测试数据增加一维平方值特征
    X_new= transform(X12)
    pred = model.predict(X_new)
    y_pred = pred.reshape(X1.shape)
    cmap = ListedColormap(['yellow','lightgray'])
    ax2.contourf(X1,X2, y_pred, cmap=cmap)
    # 原始样本数据
    show_samples(ax2, X_raw, Y)

    plt.show()

# 显示样本
def show_samples(ax, X, Y):
    for i in range(Y.shape[0]):
        if (Y[i] == 1):
            ax.scatter(X[i,0], X[i,1], marker='^', color='r')
        else:
            ax.scatter(X[i,0], X[i,1], marker='o', color='b')
        ax.text(X[i,0]+0.1, X[i,1]+0.1, str(i))

# 把数据从一维变换到二维(x2_new = x1*x1 + x2_origin)
def transform(X_raw):
    X = np.zeros_like(X_raw)
    X[:,0] = X_raw[:,0]
    X[:,1] = X_raw[:,0] ** 2 + X_raw[:,1]
    return X

if __name__=="__main__":
    X_raw = np.array([[-1.5,0], [-1,0], [-0.5,0], [0,0], [0.5,0], [1,0], [1.5,0]])
    Y = np.array([-1,-1,1,1,1,-1,-1])

    X = transform(X_raw)
    model = linear_svc(X, Y)
    show_result(model, X_raw, X, Y)
