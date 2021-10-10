
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from numpy.linalg.linalg import norm
from sklearn.datasets import *
from sklearn.svm import *
from sklearn.pipeline import *
from sklearn.preprocessing import *

def draw_circle_3d(ax, x, y):

    ax.scatter(x[y==1,0], x[y==1,1], x[y==1,2], marker='.')
    ax.scatter(x[y==0,0], x[y==0,1], x[y==0,2], marker='^')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")

def draw_2d(ax, x, y):
    ax.scatter(x[y==1,0], x[y==1,1], marker='.')
    ax.scatter(x[y==0,0], x[y==0,1], marker='^')

def svc(ax, C, X, Y):

    #X_new = X
    model = SVC(kernel='poly', degree=3, coef0=1)
    model.fit(X,Y)

    print(model)
    print(model.n_support_)
    #print(model.support_vectors_)

    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    X1,X2 = np.meshgrid(x1,x2)
    X12 = np.c_[X1.ravel(), X2.ravel()]
    pred = model.predict(X12)
    y_pred = pred.reshape(X1.shape)
    ax.contourf(X1,X2, y_pred)


    draw_2d(ax, X, y)
    ax.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], marker='o', color='', edgecolors='r', s=200)


    ax.show()


def gaussian_kernal(X, y):
    x1 = X[:,0:1]
    x2 = X[:,1:2]
    a = x1*x1 + x2*x2
    b = (np.linalg.norm(X, axis=1)).reshape(-1,1)
    print(np.allclose(a,b))

    z1 = np.exp(-a)
    z2 = np.exp(-a) * b
    z3 = np.exp(-a) * b**2
    z4 = np.exp(-a) * b**3
    z5 = np.exp(-a) * b**4
    x = np.hstack((z1,z2,z3,z4,z5))
    return x

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x[y==1,0], x[y==1,1], x[y==1,2], marker='^')
    ax.scatter(x[y==0,0], x[y==0,1], x[y==0,2], marker='o')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")

    plt.show()


if __name__=="__main__":

    X,y = make_moons(n_samples=10, noise=0.1)
    ss = StandardScaler()
    X_new = ss.fit_transform(X)
    x = gaussian_kernal(X_new,y)


    model = SVC(C=1, kernel='rbf', degree=3)
    model.fit(x,y)

    #print("权重:",model.coef_)
    print("支持向量个数:",model.n_support_)
    print("支持向量索引:",model.support_)
    print("支持向量:",model.support_vectors_)
    print("支持向量ay:",model.dual_coef_)
    print("准确率:", model.score(x, y))

    


    exit(0)

    ss = StandardScaler()
    X_new = ss.fit_transform(X)


    svc(plt, 1, X_new, y)


    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax1.axis('equal')
    ax1.scatter(X_new[y==1,0], X_new[y==1,1], marker='.')
    ax1.scatter(X_new[y==0,0], X_new[y==0,1], marker='^')
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")

    x1 = X_new[:,0].reshape(-1,1)
    x2 = X_new[:,1].reshape(-1,1)

    z = np.sqrt(x1**2 + x2**2)
    z1 = z ** 3
    z2 = z ** 2
    z3 = z

    X = np.hstack((z1,z2,z3))

    ax1 = fig.add_subplot(132)
    ax1.axis('equal')
    ax1.scatter(X_new[y==1,0], X_new[y==1,1]-1, marker='.')
    ax1.scatter(X_new[y==0,0], X_new[y==0,1]+1, marker='^')
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")



    ax2 = fig.add_subplot(133, projection='3d')
    draw_circle_3d(ax2, X, y)

    plt.show()
