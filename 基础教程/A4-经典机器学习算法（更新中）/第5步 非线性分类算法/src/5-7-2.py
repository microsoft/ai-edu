from numpy.core.defchararray import mod
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt


def svc(ax, C, X, Y):
    model = SVC(C, kernel='linear')
    model.fit(X,Y)

    print("w =", model.coef_[0])
    print("b =", model.intercept_)
    print("支持向量样本序号: ", model.support_)
    print("支持向量样本值: ", model.support_vectors_)
    
    print("support vector num: ", model.n_support_)
    print("distance: ", model.decision_function(X))
    print("alpha * y:", model.dual_coef_)

    ax.axis('equal')
    ax.grid()
    

    w = model.coef_[0]
    a = -w[0]/w[1]
    b = model.intercept_
    x = np.linspace(0,5,10)
    y0 = a * x + -b/w[1]
    y1 = a * x + -b/w[1] + 1/w[1]
    y2 = a * x + -b/w[1] - 1/w[1]
    ax.plot(x,y0)
    ax.plot(x,y1,linestyle='--')
    ax.plot(x,y2,linestyle='--')

    for i in range(Y.shape[0]):
        if (Y[i] == 1):
            ax.scatter(X[i,0], X[i,1], marker='x', color='r')
        else:
            ax.scatter(X[i,0], X[i,1], marker='.', color='b', s=200)
        ax.text(X[i,0]+0.1, X[i,1]+0.1, str(i))

    ax.set_title(str.format("C={0},W1={1:.1f},W2={2:.1f},b={3:.1f}", C, w[0], w[1], b[0]))

def main():
    X = np.array([[3,0],[2,4],[3,4],[3,3],[3,2],[3,1],[1,1],[2,2],[2,1],[2,0],[4,1]])
    Y = np.array([1,1,1,1,1,1,-1,-1,-1,-1,-1])

    fig = plt.figure()
    ax = fig.add_subplot(121)
    C = 10
    svc(ax, C, X, Y)
    ax = fig.add_subplot(122)
    C = 1
    svc(ax, C, X, Y)
    plt.show()

if __name__=="__main__":
    main()
