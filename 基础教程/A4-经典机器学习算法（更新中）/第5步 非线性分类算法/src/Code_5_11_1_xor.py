
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.preprocessing import *
import sys
import os
from pathlib import Path
from sklearn.svm import *
import matplotlib.cm as cm

def gaussian_fun2(gamma, center_x, center_y, y_i):
    xx = np.linspace(-2,2,100)
    yy = np.linspace(-2,2,100)
    P,Q = np.meshgrid(xx, yy)
    R = np.exp(-gamma * ((P-center_x)**2/2 + (Q-center_y)**2/2)) * y_i
    #ax.plot_surface(P, Q, R, cmap=cm.coolwarm)
    return P,Q,R
    #cset = ax.contour(P,Q,R,zdir='z',offset=0,cmap=cm.coolwarm)                     #绘制xy面投影


def K2(X, L, gamma):
    print("gamma=", gamma)
    n = X.shape[0]  # sample number
    m = L.shape[0]  # feature number
    K = np.zeros(shape=(n,m))
    for i in range(n):
        for j in range(m):
            K[i,j] = np.exp(-gamma * np.linalg.norm(X[i] - L[j])**2)

    #print("kernal matrix")
    #print(np.round(K,3))
    return K


def K(X, gamma):
    print("gamma=", gamma)
    n = X.shape[0]
    K = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = np.exp(-gamma * np.linalg.norm(X[i] - X[j])**2)

    print("kernal matrix")
    print(np.round(K,3))
    return K


def generate_file_path(file_name):
    curr_path = sys.argv[0]
    curr_dir = os.path.dirname(curr_path)
    file_path = os.path.join(curr_dir, file_name)
    return file_path



def test():
    gamma = 0.1
    x_0 = np.array([[0,0]])
    x_i = np.array([[1,1],[2,0],[2,2]])
    for i in range(x_i.shape[0]):
        K = np.exp(-gamma * np.linalg.norm(x_i[i] - x_0)**2)
        print(K)


def test2():
    sigma = 1.5
    X = np.linspace(-3,7,11)
    #X = np.linspace(-5,5,11)
    sum = 0
    for x in X:
        a = np.exp(-(x**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
        b = np.exp(-((x-2)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
        print(x,a,b)
        sum += min(a,b)
        #sum += a
    print(sum)

def gaussian_2d(x, sigma, mu):
    a = - (x-mu)**2 / (2*sigma*sigma)
    f = np.exp(a) / (sigma * np.sqrt(2*np.pi))
    return f

def test3():
    sigma = 1
    x = np.linspace(-4,6,100)
    f1 = gaussian_2d(x, sigma, 1)
    f2 = gaussian_2d(x, sigma, 2)
    f3 = np.exp(-(2*x*x - 6*x + 5)/(2*sigma*sigma)) / (sigma ** 2 * 2*np.pi)
    print(np.sum(f1), np.sum(f2), np.sum(f3))
    fig = plt.figure()
    plt.plot(x, f1)
    plt.plot(x, f2)
    plt.plot(x, f3)
    plt.plot(x, f1 * f2)
    plt.grid()
    plt.show()






def linear_svc(X,Y):
    #model = SVC(C=3, kernel='poly', degree=2, gamma=1, coef0=1)
    model = SVC(C=3, kernel='linear')
    model.fit(X,Y)

    #print("权重:",model.coef_)
    print("支持向量个数:",model.n_support_)
    print("支持向量索引:",model.support_)
    print("支持向量:",np.round(model.support_vectors_,3))
    print("支持向量ay:",model.dual_coef_)
    print("准确率:", model.score(X, Y))

    return model

def poly_svc(X,Y):
    model = SVC(C=3, kernel='poly', degree=2, gamma=1, coef0=1)
    model.fit(X,Y)

    #print("权重:",model.coef_)
    print("支持向量个数:",model.n_support_)
    print("支持向量索引:",model.support_)
    print("支持向量:",np.round(model.support_vectors_,3))
    print("支持向量ay:",model.dual_coef_)
    print("准确率:", model.score(X, Y))

    return model


def show_result(model, X_sample, Y):

    fig = plt.figure()

    x1 = np.linspace(-1.5, 1.5, 10)
    x2 = np.linspace(-1.5, 1.5, 10)
    X1,X2 = np.meshgrid(x1,x2)
    X12 = np.c_[X1.ravel(), X2.ravel()]
    X12_new = mapping_function(X12, 2)
    pred = model.predict(X12_new)
    y_pred = pred.reshape(X1.shape)
    plt.contourf(X1,X2, y_pred)

    draw_2d(plt, X_sample, Y)

    plt.show()


def draw_2d(ax, x, y):
    ax.scatter(x[y==1,0], x[y==1,1], marker='^')
    ax.scatter(x[y==-1,0], x[y==-1,1], marker='o')

def show_samples(X_raw, X, Y):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.grid()
    ax1.set_xlim((-1.5,1.5))
    ax1.set_ylim((-1.5,1.5))
    ax1.axis('equal')
    draw_2d(ax1, X_raw, Y)
    ax2 = fig.add_subplot(122)
    ax2.grid()
    ax2.set_xlim((-1.5,1.5))
    ax2.set_ylim((-1.5,1.5))
    ax2.axis('equal')
    draw_2d(ax2, X, Y)
    plt.show()


# 理论上的映射函数，但实际上不能用
def mapping_function(X, gamma):
    n = X.shape[0]
    Z = np.zeros((n, 4))    # 做一个4维的特征映射，即式10中的n=0,1,2,3
    for i in range(n):
        # 求 x 矢量的模，是一个标量
        x_norm = np.linalg.norm(X[i])
        # 第 0 维
        Z[i,0] = np.exp(-gamma * (x_norm**2))
        # 第 1 维
        Z[i,1] = np.sqrt(2) * x_norm * Z[i,0]
        # 第 2 维
        Z[i,2] = np.sqrt(2**2/2) * (x_norm**2) * Z[i,0]
        # 第 3 维
        Z[i,3] = np.sqrt(2**3/6) * (x_norm**3) * Z[i,0]

    return Z


def load_data(file_name):
    file_path = generate_file_path(file_name)
    file = Path(file_path)
    if file.exists():
        samples = np.loadtxt(file, delimiter=',')
        X = samples[:, 0:2]
        Y = samples[:, 2]
    else:
        X = np.random.randn(200,2)
        Y_01 = np.logical_xor(X[:,0] > 0, X[:,1] > 0)
        Y = np.zeros(Y_01.shape)
        Y[Y_01 == False] = -1
        Y[Y_01 == True] = 1
        samples = np.hstack((X,Y.reshape(-1,1)))
        np.savetxt(file_path, samples, fmt='%f, %f, %d', delimiter=',', header='x1, x2, y')
    return X, Y

if __name__=="__main__":
    # 生成原始样本
    X_raw = np.array([[0,0],[1,1],[0,1],[1,0]])
    Y = np.array([-1,-1,1,1])
    print("X 的原始值：")
    print(X_raw)
    print("Y 的原始值：")
    print(Y)
    
    # 标准化
    ss = StandardScaler()
    X = ss.fit_transform(X_raw)
    print("X 标准化后的值：")
    print(X)

    # X 标准化后映射的特征值
    gamma = 2
    Z = mapping_function(X, gamma)
    print("X 标准化后映射的特征值：")
    print(Z)
    # 通过结果可以看出来映射后4个样本被映射到了四维空间中的一个点，不能做后续的分类
   
    # X 不做标准化直接做映射的特征值
    gamma = 2
    Z = mapping_function(X_raw, gamma)
    print("X 不做标准化直接做映射的特征值：")
    print(Z)

    # 用 K 函数做映射，形成核函数矩阵



    # 尝试用线性 SVM 做分类    
    model = linear_svc(Z, Y)
    


