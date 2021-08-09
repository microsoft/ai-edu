import numpy as np

x = np.array([1,2,3,4,5,6])
p = np.array([0.1,0.1,0.1,0.3,0.4,0.0])

def mean(x):
    return np.sum(x)/x.shape[0]

def E(x, p):
    return np.sum(x*p)

def var(x):
    mu = mean(x)
    return np.sum((x-mu)*(x-mu))/x.shape[0]

def S(x):
    mu = mean(x)
    return np.sum((x-mu)*(x-mu))/(x.shape[0]-1)

def D(x,p):
    expect = E(x,p)
    a = (x - expect) * (x - expect)
    print(E(a,p))

    print(E(x*x,p) - E(x,p)*E(x,p))


if __name__ == '__main__':
    print("mean=", mean(x))
    print("ex=", E(x,p))
    print("var=", var(x))
    print("S2=", S(x))
    print("D=", D(x,p))
    
