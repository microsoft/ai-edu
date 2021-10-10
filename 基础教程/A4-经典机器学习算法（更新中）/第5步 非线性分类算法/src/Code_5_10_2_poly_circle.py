
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.svm import SVC
from sklearn.preprocessing import *

def draw_circle_3d(ax, x, p, n):
    ax.scatter(x[p,0], x[p,1], x[p,2], marker='.')
    ax.scatter(x[n,0], x[n,1], x[n,2], marker='^')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")

def draw_2d(ax, x, pos, neg):
    ax.scatter(x[pos,0], x[pos,1], marker='.')
    ax.scatter(x[neg,0], x[neg,1], marker='^')


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






if __name__=="__main__":

    test()
    exit(0)

    x,y = make_circles(n_samples=100, factor=0.5, noise=0.1)
    svc(plt, 1, x, y)
    exit(0)
    #x,y = make_moons(n_samples=100, noise=0.1)

    

    pos = (y == 1)
    neg = (y == 0)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax1.axis('equal')
    draw_2d(ax1, x, pos, neg)
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")

    # f(z)=[z1*z1, z1*z2, z2*z2]
    x1 = (x[:,0]**2).reshape(-1,1)
    x2 = (1.414 * x[:,0] * x[:,1]).reshape(-1,1)
    x3 = (x[:,1]**2).reshape(-1,1)
    X = np.hstack((x1,x3,x2))
    print(X)

    ax2 = fig.add_subplot(132)
    ax2.axis('equal')
    draw_2d(ax2, X, pos, neg)

    ax3 = fig.add_subplot(133, projection='3d')
    draw_circle_3d(ax3, X, pos, neg)

    plt.show()
