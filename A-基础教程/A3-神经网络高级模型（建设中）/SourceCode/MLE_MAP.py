import numpy as np
import math
from scipy.special import comb,perm
import matplotlib.pyplot as plt

class CheckMaximumL(object):
    def __init__(self):
        self.p = 0
        self.l = 0

    def set(self, p, l):
        if (l > self.l):
            self.p = p
            self.l = l

    def get(self):
        return self.p, self.l

def strf(prefix, f):
    return str.format("{0}={1:2f}", prefix, f)

def BernoulliDist():
    L = []
    N = 10
    for x in range(N+1):
        l = comb(N,x) * math.pow(0.6,x) * math.pow(0.4,N-x)
        L.append(l)
        plt.text(x,l,np.around(l, 3))
    print(np.sum(L))

    plt.plot(L, '-o')
    plt.xlabel("x")
    plt.ylabel("probability")
    plt.title("Bernoulli Distribution")
    plt.grid()
    plt.show()

def Likelihood_bernoulli(N,X,title="Bernoulli Likelihood"):
    checker = CheckMaximumL()
    L = []
    P = np.linspace(0,1,num=101)
    for p in P:
        l = math.pow(p,X) * math.pow(1-p,N-X)
        L.append(l)
        checker.set(p, l)
    #endfor
    plt.plot(P, L)
    x,y = checker.get()
    plt.plot(x,y,'x')
    plt.text(x,y,strf(r'$\theta$', x))
    plt.xlabel(r'$\theta$')
    plt.ylabel("likelihood")
    plt.title(title)
    plt.grid()
    plt.show()


def likelihood_exp(X, style):
    checker = CheckMaximumL()
    L = []
    P = np.linspace(0.01,3,num=100)
    n = X.shape[0]
    mean = np.mean(X)
    print(mean)
    for p in P:
        l = np.exp(-np.sum(X)/p) / np.power(p,n)
        l = l * np.power(X[0],3)    # 不是公式的一部分，是为了让两条线的高度差不多
        L.append(l)
        checker.set(p,l)
    ll, = plt.plot(P, L, linestyle=style)
    x,y = checker.get()
    plt.plot(x,y,'x')
    plt.text(x,y,strf(r'$\theta$', mean))
    plt.xlabel(r'$\theta$')
    plt.ylabel("likelihood")
    return ll
    
def exp_fun(p):
    n = 10
    x = np.linspace(0,1,num=n)
    X = 1 / p * np.exp(-1 * x / p)
    return X

def likelihood_exps():
    lines = []
    labels = []
    styles = ['-', '--','-.',':']
    p = [0.5, 0.3]
    for i in range(2):
        X = exp_fun(p[i])
        print(X)
        print(np.mean(X))
        ll = likelihood_exp(X, styles[i])
    plt.title("Exponential Distribution")
    plt.grid()
    plt.show()

def likelihood_norm_mu(style):
    checker = CheckMaximumL()
    L = []
    P = np.linspace(0,1,num=101)
    n=10
    X = np.random.normal(0.5,0.5,n)
    mu_max = np.mean(X)
    std = np.std(X)
    for mu in P:
        sum = np.sum(np.square(X - mu))
        l = np.exp(-0.5 * sum / np.square(std)) * np.power(np.sqrt(2*np.pi*std), n)
        L.append(l)
        checker.set(mu, l)
    ll, = plt.plot(P, L, linestyle=style)
    x,y = checker.get()
    plt.plot(x,y,'x')
    plt.text(x,y,strf(r'$\mu$', mu_max))
    plt.xlabel(r'$\mu$')
    plt.ylabel("likelihood")
    return ll

def likelihood_norm_sigma(style):
    checker = CheckMaximumL()
    L = []
    P = np.linspace(0.01,1,num=100)
    n=10
    X = np.random.normal(0.5,0.5,n)
    mu = np.mean(X)
    std_max = np.sum(np.square(X-mu))/n
    for sigma2 in P:
        l = np.exp(-0.5 * np.sum(np.square(X-mu)) / sigma2)  / np.power(np.sqrt(sigma2*2*np.pi),n)
        L.append(l)
        checker.set(sigma2, l)
    ll, = plt.plot(P, L, linestyle=style)
    x,y = checker.get()
    plt.plot(x,y,'x')
    plt.text(x,y,strf(r'$\sigma^2$', std_max))
    plt.xlabel(r'$\sigma^2$')
    plt.ylabel("likelihood")
    #plt.show()
    return ll

def likelihood_norms():
    styles = ['-', '--','-.',':']
    for i in range(2):
        likelihood_norm_mu(styles[i])
    plt.title("Normal Distribution")
    plt.grid()
    plt.show()

    styles = ['-', '--','-.',':']
    for i in range(2):
        likelihood_norm_sigma(styles[i])
    plt.title("Normal Distribution")
    plt.grid()
    plt.show()

def BetaDist(a, b, style):
    c = math.factorial(a+b-1)/(math.factorial(a-1)*math.factorial(b-1))
    F = []
    X = np.linspace(0,1,num=101)
    for x in X:
        f = c * math.pow(x, a-1) * math.pow(1-x, b-1)
        F.append(f)
    #endfor
    l = str.format(r"$\alpha={0}, \beta={1}$", a, b)
    #print(l)
    plt.plot(X,F,label=l,linestyle=style)

def likelihood_Beta(alpha, beta, a, b):
    checker = CheckMaximumL()
    c = math.factorial(alpha+beta-1)/(math.factorial(alpha-1)*math.factorial(beta-1))
    F = []
    X = np.linspace(0,1,num=101)
    for x in X:
        f = c * math.pow(x, a) * math.pow(1-x, b)
        F.append(f)
        checker.set(x, f)
    #endfor
    x,y = checker.get()
    plt.plot(x,y,'x')
    plt.text(x,y,strf(r'$\theta$', x))
    plt.plot(X,F)
    plt.grid()
    plt.title("Beta Distribution")
    plt.show()

def BetaDists():
    BetaDist(3,3,'-')
    BetaDist(5,1,'-.')
    BetaDist(2,2,'--')
    BetaDist(1,3,':')
    plt.grid()
    plt.ylim(ymax=2.5)
    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.title("Beta Distribution")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    
    BernoulliDist()
    Likelihood_bernoulli(N=10,X=7)
    likelihood_exps()
    likelihood_norms()
    
    BetaDists()
    
    
    likelihood_Beta(3, 3, 9, 5)



