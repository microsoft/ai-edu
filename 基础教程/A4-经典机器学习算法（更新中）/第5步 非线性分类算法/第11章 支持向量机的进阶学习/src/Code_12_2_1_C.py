from numpy.core.defchararray import mod
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt


def show_samples(ax, X, Y):
    for i in range(Y.shape[0]):
        if (Y[i] == 1):
            ax.scatter(X[i,0], X[i,1], marker='^', color='r')
        else:
            ax.scatter(X[i,0], X[i,1], marker='o', color='b')
        ax.text(X[i,0]+0.1, X[i,1]+0.1, str(i))

def svc(ax, C, X, Y):
    model = SVC(C, kernel='linear')
    model.fit(X,Y)

    print("w =", model.coef_[0])
    print("b =", model.intercept_)
    print("支持向量样本序号: ", model.support_)
    print("支持向量样本值: ", model.support_vectors_)
    print("支持向量数量: ", model.n_support_)
    print("到分界线的距离: ", model.decision_function(X))
    print("alpha * y:", model.dual_coef_)

    ax.axis('equal')
    ax.grid()
    
    x = np.linspace(0,5,10)
    w = model.coef_[0]
    b = model.intercept_[0]

    # w[0] * x[0] + w[1] * x[1] + b = 0
    y0 = (-w[0] * x - b)/w[1]
    # w[0] * x[0] + w[1] * x[1] + b = 1
    y1 = (-w[0] * x - b + 1)/w[1]
    # w[0] * x[0] + w[1] * x[1] + b = -1
    y2 = (-w[0] * x - b - 1)/w[1]

    ax.plot(x,y0)
    ax.plot(x,y1,linestyle='--')
    ax.plot(x,y2,linestyle='--')

    show_samples(ax, X, Y)

    ax.set_title(str.format("C={0},W1={1:.2f},W2={2:.2f},b={3:.1f}", C, w[0], w[1], b/(-w[1])))

def main():
    X = np.array([[0,3],[1,1],[1,2],[2,1],[3,0],[1,3],  [1,4],[2,3],[3,1],[3,2],[3,3],[4,1]])
    Y = np.array([-1,-1,-1,-1,-1,-1, 1,1,1,1,1,1])

    fig = plt.figure()
    plt.grid()
    plt.axis('equal')
    show_samples(plt, X, Y)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(131)
    C = 10
    svc(ax, C, X, Y)
    ax = fig.add_subplot(132)
    C = 1
    svc(ax, C, X, Y)
    ax = fig.add_subplot(133)
    C = 0.1
    svc(ax, C, X, Y)
    plt.show()

if __name__=="__main__":
    main()
