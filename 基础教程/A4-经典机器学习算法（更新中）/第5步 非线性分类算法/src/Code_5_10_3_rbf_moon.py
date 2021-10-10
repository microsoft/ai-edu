
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




def K(X, gamma):
    print("gamma=", gamma)

    K = np.zeros(shape=(10,10))
    for i in range(10):
        for j in range(10):
            K[i,j] = np.exp(-gamma * np.linalg.norm(X[i] - X[j])**2)

    print("kernal matrix")
    print(np.round(K,2))

    '''
    Z = np.zeros(shape=(10,10))
    for i in range(10):
        for j in range(10):
            Z[i,j] = np.inner(K[i], K[j])
    print("inner Z")
    print(np.round(Z,2))
    '''

def generate_file_path(file_name):
    curr_path = sys.argv[0]
    curr_dir = os.path.dirname(curr_path)
    file_path = os.path.join(curr_dir, file_name)
    return file_path

def draw_2d(ax, x, y):
    ax.scatter(x[y==1,0], x[y==1,1], marker='^')
    ax.scatter(x[y==-1,0], x[y==-1,1], marker='o')


def svc(ax, C, X, Y):

    model = SVC(kernel='rbf', gamma=2)
    model.fit(X,Y)

    print("支持向量个数:",model.n_support_)
    print("支持向量索引:",model.support_)
    print("支持向量:",model.support_vectors_)
    print("支持向量ay:",model.dual_coef_)

    x1 = np.linspace(-2, 2, 100)
    x2 = np.linspace(-2, 2, 100)
    X1,X2 = np.meshgrid(x1,x2)
    X12 = np.c_[X1.ravel(), X2.ravel()]
    pred = model.predict(X12)
    y_pred = pred.reshape(X1.shape)
    ax.contourf(X1,X2, y_pred)

    draw_2d(ax, X, Y)
    ax.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], marker='o', color='', edgecolors='r', s=200)

    return model

if __name__=="__main__":
    file_name = "5-0-moon-data.csv"
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

    ss = StandardScaler()
    X = ss.fit_transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.grid()

    model = svc(ax, 1, X, Y)
    #model = svc(ax, 1, X, Y=[1,1,1,1,1,0,0,0,0,0])
    xx = np.array([[0,0],[0.5,0],[1.5,1.5],[1.5,-1.5],[-1.5,1.5],[-1.5,-1.5]])
    #xx = np.array([[-1.32,0.95],[-0.1,-0.9]])
    pred = model.predict(np.array(xx))
    print("------", pred)

    gamma = 2

    for j in range(xx.shape[0]):
        sum = 0
        for i in range(10):
            sum += model.dual_coef_[0,i] * np.exp(-np.linalg.norm(xx[j] - model.support_vectors_[i])**2*gamma)
        sum += model.intercept_
        yy = np.sign(sum)
        print("预测:", xx[j], sum, yy)

    
    ax2 = fig.add_subplot(122, projection='3d')
    RR = None
    for i in range(0,10):
        P,Q,R = gaussian_fun2(gamma*2, model.support_vectors_[i,0],model.support_vectors_[i,1], Y[i])
        if (RR is None):
            RR = R
        else:
            RR += R
    #ax2.plot_surface(P, Q, RR, cmap=cm.coolwarm)
    ax2.contour(P,Q,RR,zdir='z',offset=0,cmap=cm.coolwarm)                     #绘制xy面投影

    '''
    K(X, 0.5)
    K(X, 1)
    K(X, 2)
    
    ax2 = fig.add_subplot(122)
    ax2.grid()
    ax2.axis('equal')
    draw_2d(ax2, X, Y)

    for i in range(10):
        ax2.text(X[i,0], X[i,1], str(i))
        if (i <= 4):
            cir = plt.Circle((X[i,0], X[i,1]), 0.5, linestyle='--', color='r', fill=False)
            ax2.add_patch(cir)
        else:
            cir = plt.Circle((X[i,0], X[i,1]), 0.5, linestyle='--', color='b', fill=False)
            ax2.add_patch(cir)
    '''
    plt.show()


