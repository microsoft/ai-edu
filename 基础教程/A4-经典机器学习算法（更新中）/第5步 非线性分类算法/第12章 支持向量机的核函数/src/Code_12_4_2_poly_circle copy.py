
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.svm import SVC
from sklearn.preprocessing import *
import sys
import os
from pathlib import Path
import matplotlib as mpl

def K2(X, L, gamma):
    print("gamma=", gamma)
    n = X.shape[0]  # sample number
    m = L.shape[0]  # feature number
    K = np.zeros(shape=(n,m))
    for i in range(n):
        for j in range(m):
            K[i,j] = np.exp(-gamma * np.linalg.norm(X[i] - L[j])**2)

    print("kernal matrix")
    print(np.round(K,3))
    return K

def sigmoid(x):
    a = 1.0 / (1.0 + np.exp(-x))
    return a

def K3(X, L, gamma):
    print("gamma=", gamma)
    n = X.shape[0]  # sample number
    m = L.shape[0]  # feature number
    K = np.zeros(shape=(n,m))
    for i in range(n):
        for j in range(m):
            K[i,j] =  1 - sigmoid(-gamma * np.linalg.norm(X[i] - L[j])**2)

    print("kernal matrix")
    print(np.round(K,3))
    return K



def svc(ax, C, X, Y):
    model = SVC(C=1, kernel='poly', degree=2)   # d=2,4,6...均可求解
    model = SVC(C=1, kernel='rbf')   # d=2,4,6...均可求解
    model.fit(X,Y)

    print(model)

    x1 = np.linspace(-1.2, 1.2, 100)
    x2 = np.linspace(-1.2, 1.2, 100)
    X1,X2 = np.meshgrid(x1,x2)
    X = np.c_[X1.ravel(), X2.ravel()]
    pred = model.predict(X)
    y_pred = pred.reshape(X1.shape)
    ax.contourf(X1,X2, y_pred)

    pos = (y == 1)
    neg = (y == 0)

    draw_2d(ax, x, pos, neg)
    ax.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], marker='o', color='', edgecolors='r', s=200)

    ax.show()

    return



def test():
    x,y = make_circles(n_samples=100, factor=0.5, noise=0.1)
    ss = StandardScaler()
    X = ss.fit_transform(x)
    X_new = K3(X, X, 2)
    model = SVC(C=1, kernel='linear')
    model.fit(X_new,y)

    print("权重:",model.coef_)
    print("支持向量个数:",model.n_support_)
    print("支持向量索引:",model.support_)
    print("支持向量:",model.support_vectors_)
    print("支持向量ay:",model.dual_coef_)
    print("准确率:", model.score(X_new, y))

    return model

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

def draw_circle_3d(ax, x, y):
    ax.scatter(x[y==1,0], x[y==1,1], x[y==1,2], marker='^', c='r')
    ax.scatter(x[y==-1,0], x[y==-1,1], x[y==-1,2], marker='o', c='b')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")

def draw_2d(ax, x, y):
    ax.scatter(x[y==1,0], x[y==1,1], marker='^', c='r')
    ax.scatter(x[y==-1,0], x[y==-1,1], marker='o', c='b')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

if __name__=="__main__":

    X_raw, Y = load_data("Data_11_4_Circle_100.csv")
   
    fig = plt.figure()
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    
    # 原始样本
    ax1 = fig.add_subplot(131)
    ax1.axis('equal')
    ax1.grid()
    ax1.set_title(u"原始样本数据")
    draw_2d(ax1, X_raw, Y)
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    
    # 重新构建二维特征
    ax2 = fig.add_subplot(132)
    ax2.axis('equal')
    ax2.grid()
    ax2.set_title(u"重新构建二维特征")
    
    # 前两维数据都变成自身的平方，不增加新维
    X_2d = np.zeros_like(X_raw)
    X_2d[:,0] = X_raw[:,0]**2
    X_2d[:,1] = X_raw[:,1]**2
    draw_2d(ax2, X_2d, Y)
  
    # 构建三维特征
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title(u"构建三维特征")
    
    # 前两维特征不变
    X_3d = np.zeros((X_raw.shape[0], 3))
    X_3d[:,0:2] = X_raw
    # 增加一维
    X_3d[:,2] = X_raw[:,0]**2 + X_raw[:,1]**2
    draw_circle_3d(ax3, X_3d, Y)

    plt.show()
