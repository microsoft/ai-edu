
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.svm import SVC
from sklearn.preprocessing import *
import matplotlib as mpl

# 线性SVM分类器
def linear_svc(X,Y):
    model = SVC(C=10, kernel='linear')
    model.fit(X,Y)

    print("权重:",model.coef_)
    print("偏移:",model.intercept_)
    print("支持向量个数:",model.n_support_)
    print("支持向量索引:",model.support_)
    print("支持向量:",np.round(model.support_vectors_,3))
    print("支持向量ay:",model.dual_coef_)
    print("准确率:", model.score(X, Y))

    return model


def show_samples_3d(ax, X, Y):
    for i in range(Y.shape[0]):
        if (Y[i] == 1):
            ax.scatter(X[i,0], X[i,1], X[i,2], marker='^', color='r')
        else:
            ax.scatter(X[i,0], X[i,1], X[i,2], marker='o', color='b')

        ax.text(X[i,0]+0.05, X[i,1], X[i,2], str(i))

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")

def show_samples(ax, X, Y):
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    for i in range(Y.shape[0]):
        if (Y[i] == 1):
            ax.scatter(X[i,0], X[i,1], marker='^', color='r')
        else:
            ax.scatter(X[i,0], X[i,1], marker='o', color='b')
        ax.text(X[i,0]+0.05, X[i,1], str(i))

if __name__=="__main__":

    X_raw = np.array([[0,0],[1,1],[0,1],[1,0]])
    Y = np.array([-1,-1,1,1])

    X = np.zeros((X_raw.shape[0], 3))
    X[:,0] = X_raw[:,0]
    X[:,1] = X_raw[:,1]
    X[:,2] = X_raw[:,0] * X_raw[:,1]

    fig = plt.figure()
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False

    ax1 = fig.add_subplot(121)
    ax1.axis('equal')
    ax1.grid()
    ax1.set_title(u'原始样本')
    show_samples(ax1, X, Y)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title(u'新特征样本')
    show_samples_3d(ax2, X, Y)
    plt.show()

    model = linear_svc(X, Y)
    result = np.dot(model.coef_, X.T) + model.intercept_
    print("分类结果判别：", np.round(result,3))