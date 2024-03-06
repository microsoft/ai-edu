import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
mpl.rcParams['axes.unicode_minus']=False


def f(x):
    return 2.0 / (1 + np.exp(-x)) - 1

def f_normal(x, mu, sigma):
    y = 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sigma**2))
    return y

def integral(n, mu, sigma):
    X = np.random.normal(loc=mu, scale=sigma, size=(n,))
    Y = f(X)
    V = np.sum(Y) / n
    return X, V

def draw(n, mu_p, sigma_p, mu_q, sigma_q):
    x = np.linspace(-5, 5, n)    
    y = f(x)
    plt.plot(x, y, linestyle="-")
    y = f_normal(x, mu_p, sigma_p)
    plt.plot(x, y, linestyle="-.")
    y = f_normal(x, mu_q, sigma_q)
    plt.plot(x, y, linestyle=":")
    plt.grid()
    plt.legend(["f(x)=sigmoid(x)*2-1","正态分布p","正态分布q"])
    plt.show()

def P_Q(n, mu_p, sigma_p, mu_q, sigma_q):
    X = np.random.normal(loc=0, scale=sigma_p, size=(n,))
    Xp = X + mu_p
    Yp = f(Xp)
    Vp = np.sum(Yp) / n
    Xq = X + mu_q
    Yq = f(Xq)
    Vq = np.sum(Yq) / n
    p = f_normal(Xq, mu_p, sigma_p)
    q = f_normal(Xq, mu_q, sigma_q)
    W = p / q
    Y = Yq * W
    Vpq = np.sum(Y) / np.sum(W)
    #Vpq = np.sum(Y) / n
    return Vp, Vq, Vpq
    
if __name__=="__main__":
    n = 10000
    mu_p, sigma_p = 2.0, 2.0
    mu_q, sigma_q = -2.0, 2.0
    draw(n, mu_p, sigma_p, mu_q, sigma_q)

    print("分布p\t分布q\t重要性采样p/q")
    print("----------------------------------")
    VP, VQ, VPQ = None, None, None
    repeat = 10
    for i in range(repeat):
        Vp, Vq, Vpq = P_Q(n, mu_p, sigma_p, mu_q, sigma_q)
        print("{0:1.3f}\t{1:1.3f}\t{2:1.3f}".format(Vp, Vq, Vpq))
        if VP is None:
            VP, VQ, VPQ = Vp, Vq, Vpq
        else:
            VP += Vp
            VQ += Vq
            VPQ += Vpq
    print("----------------------------------")
    print("{0:1.3f}\t{1:1.3f}\t{2:1.3f}".format(VP/repeat, VQ/repeat, VPQ/repeat))
