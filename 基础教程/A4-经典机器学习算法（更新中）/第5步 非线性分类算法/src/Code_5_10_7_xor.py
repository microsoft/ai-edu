
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

def draw_2d(ax, x, y):
    ax.scatter(x[y==1,0], x[y==1,1], marker='^')
    ax.scatter(x[y==-1,0], x[y==-1,1], marker='o')


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


def svc(C, X, Y):

    model = SVC(C=C, kernel='linear')
    model.fit(X,Y)

    print("权重:",model.coef_)
    print("支持向量个数:",model.n_support_)
    print("支持向量索引:",model.support_)
    print("支持向量:", np.round(model.support_vectors_,3))
    print("支持向量ay:",model.dual_coef_)
    print("准确率:", model.score(X, Y))

    return model


def test_feature_linear(X,Y):
    gamma = 2
    K_10_10 = K(X, gamma)
    C = 1
    model = svc(C, K_10_10, Y)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.grid()
    draw_2d(ax1, X, Y)

    ax2 = fig.add_subplot(122)
    x1 = np.linspace(-2, 2, 10)
    x2 = np.linspace(-2, 2, 10)
    X1,X2 = np.meshgrid(x1,x2)
    X12 = np.c_[X1.ravel(), X2.ravel()]
    X12_new = K2(X12, X, gamma)
    pred = model.predict(X12_new)
    y_pred = pred.reshape(X1.shape)
    ax2.contourf(X1,X2, y_pred)

    plt.show()

def test_rgb(X,Y):
    model = SVC(C=1, kernel='rbf', coef0=2)
    model.fit(X,Y)

    #print("权重:",model.coef_)
    print("支持向量个数:",model.n_support_)
    print("支持向量索引:",model.support_)
    print("支持向量:",np.round(model.support_vectors_,3))
    print("支持向量ay:",model.dual_coef_)
    print("准确率:", model.score(X, Y))

    fig = plt.figure()

    x1 = np.linspace(-2, 2, 10)
    x2 = np.linspace(-2, 2, 10)
    X1,X2 = np.meshgrid(x1,x2)
    X12 = np.c_[X1.ravel(), X2.ravel()]
    pred = model.predict(X12)
    y_pred = pred.reshape(X1.shape)
    plt.contourf(X1,X2, y_pred)
    plt.show()

if __name__=="__main__":

    X = np.array([[-1,-1],[1,1],[-1,1],[1,-1]])
    Y = np.array([-1,-1,1,1])

    ss = StandardScaler()
    X = ss.fit_transform(X)

    test_feature_linear(X, Y)
