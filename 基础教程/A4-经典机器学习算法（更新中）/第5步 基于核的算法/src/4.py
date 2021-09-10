
import numpy as np
import matplotlib.pyplot as plt


def aaa():
    fig=plt.figure()
    plt.xlabel("alpha")
    plt.ylabel("L")

    x=np.linspace(-6,6,100)
    f=x**4-50*x**2+100*x
    alpha=np.linspace(-20,120,1000)
    x_min=-4.5
    f_min=x_min**4-50*x_min**2+100*x_min
    for i in range(len(x)):
        plt.plot(alpha,x[i]**4-50*x[i]**2+100*x[i]+alpha*(-x[i]+x_min))
    plt.plot(alpha,(x_min**4-50*x_min**2+100*x_min+alpha*(-x_min-4.5)),c='k')
    plt.show()

def bbb():
    Total = np.zeros((101,101))
    X = np.linspace(-1,3,101)
    Y = X**2 - 2*X + 1
    plt.plot(X,Y)
    A = np.linspace(-3,3,101)
    count = 0
    for x in X:
        L = []
        L = x*x - 2*x + 1 + A*(x-0.5)
        """
        for a in A:
            l = x*x - 2*x + 1 + a*(x-0.5)
            L.append(l)
        """
        plt.plot(X,L,linestyle='--')
        Total[count] = L
        count += 1

    plt.xlabel("A")
    plt.ylabel("L(x,a)")
    #plt.show()

    m = np.min(Total, axis=0)
    print(m)
    print(np.max(m))
    idx = np.argmax(m)
    print(idx)
    print(6*idx/101-3)
    plt.plot(X,m)
    plt.grid()
    plt.show()

if __name__=="__main__":
    bbb()
