
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.preprocessing import *
import sys
import os
from pathlib import Path
from sklearn.svm import *
import matplotlib.cm as cm

def gaussian_fun2(gamma, center_x, center_y, ay_i):
    xx = np.linspace(-2,2,100)
    yy = np.linspace(-2,2,100)
    P,Q = np.meshgrid(xx, yy)
    R = np.exp(-gamma * ((P-center_x)**2/2 + (Q-center_y)**2/2)) * ay_i
    return P,Q,R

def gaussian_kernal(gamma, landmark, weight):
    xx = np.linspace(-2,2,100)
    yy = np.linspace(-2,2,100)
    P,Q = np.meshgrid(xx, yy)
    R = weight * np.exp(-gamma * ((P-landmark[0])**2 + (Q-landmark[1])**2))
    return P,Q,R

def K2(X, L, gamma):
    print("gamma=", gamma)
    n = X.shape[0]  # sample number
    m = L.shape[0]  # feature number
    K = np.zeros(shape=(n,m))
    for i in range(n):
        for j in range(m):
            K[i,j] = np.exp(-gamma * np.linalg.norm(X[i] - L[j])**2)

    print("kernal matrix")
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
    for i in range(x.shape[0]):
        ax.text(x[i,0], x[i,1]+0.1, str(i))

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
    print("支持向量:",np.round(model.support_vectors_, 3))
    print("支持向量ay:",model.dual_coef_)
    print("准确率:", model.score(X, Y))

    return model


def load_data(file_name):
    file_path = generate_file_path(file_name)
    file = Path(file_path)
    if file.exists():
        samples = np.loadtxt(file, delimiter=',')
        X = samples[:, 0:2]
        Y = samples[:, 2]
    else:
        X, Y = make_moons(n_samples=10, noise=0.1, shuffle=False)
        Y[Y == 0] = -1
        samples = np.hstack((X,Y.reshape(-1,1)))
        np.savetxt(file_path, samples, fmt='%f, %f, %d', delimiter=',', header='x1, x2, y')
    return X, Y


def test_g(X, Y):
    gamma = 2
    model = SVC(C=3, kernel='rbf', gamma=gamma)
    model.fit(X,Y)

    #print("权重:",model.coef_)
    print("支持向量个数:",model.n_support_)
    print("支持向量索引:",model.support_)
    print("支持向量:",np.round(model.support_vectors_, 3))
    print("支持向量ay:",model.dual_coef_)
    print("准确率:", model.score(X, Y))

    fig = plt.figure()

    ax1 = fig.add_subplot(131)
    ax1.grid()
    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    X1,X2 = np.meshgrid(x1,x2)
    X12 = np.c_[X1.ravel(), X2.ravel()]
    pred = model.predict(X12)
    y_pred = pred.reshape(X1.shape)
    plt.contourf(X1,X2, y_pred)
    draw_2d(ax1, X, Y)

    n_support = np.sum(model.n_support_)
    ax2 = fig.add_subplot(132)
    ax2.grid()

    RR = None
    for i in range(n_support):
        P,Q,R = gaussian_kernal(
            gamma, 
            model.support_vectors_[i],
            model.dual_coef_[0,i])
        if (RR is None):
            RR = R
        else:
            RR += R
    
    ax2.contour(P,Q,RR,zdir='z',offset=0,cmap=cm.coolwarm)                     #绘制xy面投影
    draw_2d(ax2, X, Y)

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(P, Q, RR, cmap=cm.coolwarm)

    plt.show()

    return model


if __name__=="__main__":

    file_name = "5-0-moon-data.csv"
    X, Y = load_data(file_name)
    ss = StandardScaler()
    X = ss.fit_transform(X)
    test_g(X,Y)
    exit(0)

    L = np.array([[1,-1],[-1,1],[-1,-1],[1,1],[0,0],[0.5,0.5],[-0.5,-0.5],[-0.5,0.5],[0.5,-0.5]])

    # gamma=2, C=2

    ss = StandardScaler()
    X = ss.fit_transform(X)
    gamma = 2
    #K_10_10 = K2(X, X, gamma)
    K_10_10 = K(X, gamma)
    C = 2
    model = svc(C, K_10_10, Y)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.grid()
    draw_2d(ax1, X, Y)

    ax2 = fig.add_subplot(122)
    ax2.grid()
    x1 = np.linspace(-2, 2, 10)
    x2 = np.linspace(-2, 2, 10)
    X1,X2 = np.meshgrid(x1,x2)
    X12 = np.c_[X1.ravel(), X2.ravel()]
    X12_new = K2(X12, X, gamma)
    pred = model.predict(X12_new)
    y_pred = pred.reshape(X1.shape)
    ax2.contourf(X1,X2, y_pred)
    draw_2d(ax2, X, Y)

    plt.show()
