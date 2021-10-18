
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import *

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


def show_result(model, X_sample, Y):

    fig = plt.figure()

    x1 = np.linspace(-2, 2, 10)
    x2 = np.linspace(0, 1, 10)
    X1,X2 = np.meshgrid(x1,x2)
    X12 = np.c_[X1.ravel(), X2.ravel()]

    pred = model.predict(X12)
    y_pred = pred.reshape(X1.shape)

    print(y_pred.shape)
    print(y_pred)

    h = np.sum(y_pred[:,0]==True) / y_pred.shape[1]
    print(h)
    
    yy_pred = np.zeros_like(y_pred)
    

    l = h - x1**2
    plt.plot(x1,l)

    #plt.contourf(X1,X2, y_pred)

    show_samples(plt, X_sample, Y)

    plt.show()


def show_samples(ax, X, Y):
    for i in range(Y.shape[0]):
        if (Y[i] == 1):
            ax.scatter(X[i,0], 0, marker='^', color='r')
        else:
            ax.scatter(X[i,0], 0, marker='o', color='b')
        ax.text(X[i,0]+0.1, 0.1, str(i))

if __name__=="__main__":
    X_raw = np.array([[-1.5], [-1], [-0.5], [0], [0.5], [1], [1.5]])
    Y = np.array([-1,-1,1,1,1,-1,-1])

    X = np.hstack((X_raw, X_raw * X_raw))
    print(X)
    model = linear_svc(X, Y)
    show_result(model,X_raw,Y)

