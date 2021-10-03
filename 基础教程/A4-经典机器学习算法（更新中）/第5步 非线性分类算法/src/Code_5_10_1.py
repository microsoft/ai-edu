import numpy as np
import time

def f(z):
    n = z.shape[1]
    Z = np.zeros(shape=(1,n*n))
    for i in range(3):
        for j in range(3):
            Z[0,i*3+j] = z[0,i]*z[0,j]
    '''
    result = np.array([[
        z[0,0]*z[0,0],z[0,0]*z[0,1],z[0,0]*z[0,2],
        z[0,1]*z[0,0],z[0,1]*z[0,1],z[0,1]*z[0,2],
        z[0,2]*z[0,0],z[0,2]*z[0,1],z[0,2]*z[0,2]
    ]])
    '''
    return Z


def fx_fy(x,y):
    fx = f(x)
    fy = f(y)
    result = np.inner(fx, fy)
    '''
    result = \
    x[0,0]*x[0,0] * y[0,0]*y[0,0] +\
    x[0,0]*x[0,1] * y[0,0]*y[0,1] +\
    x[0,0]*x[0,2] * y[0,0]*y[0,2] +\
    x[0,1]*x[0,0] * y[0,1]*y[0,0] +\
    x[0,1]*x[0,1] * y[0,1]*y[0,1] +\
    x[0,1]*x[0,2] * y[0,1]*y[0,2] +\
    x[0,2]*x[0,0] * y[0,2]*y[0,0] +\
    x[0,2]*x[0,1] * y[0,2]*y[0,1] +\
    x[0,2]*x[0,2] * y[0,2]*y[0,2]
    '''
    return result

def k(x,y):
    result = np.inner(x, y)**2
    #result = (x[0,0]*y[0,0] + x[0,1]*y[0,1]+x[0,2]*y[0,2])**2
    return result


if __name__=="__main__":
    x = np.array([1,2,3]).reshape(1,3)
    y = np.array([4,5,6]).reshape(1,3)
    n = 100000
    start = time.time()
    for i in range(n):
        r1 = fx_fy(x, y)
    end1 = time.time()
    for i in range(n):
        r2 = k(x, y)
    end2 = time.time()
    print(r1,r2)
    print(end1-start, end2-end1)
