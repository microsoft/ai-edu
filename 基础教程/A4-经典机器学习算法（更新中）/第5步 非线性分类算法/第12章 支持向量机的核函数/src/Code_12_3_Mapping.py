import numpy as np
import time

def f(z):
    n = z.shape[1]
    assert(n==3)
    Z = np.zeros(shape=(1,n*n))
    # 生成[z_1 z_1,z_1 z_2,z_1 z_3,z_2 z_1,z_2 z_2,z_2 z_3,z_3 z_1,z_3 z_2,z_3 z_3]
    for i in range(3):
        for j in range(3):
            Z[0,i*3+j] = z[0,i]*z[0,j]
    return Z

def fx_fy(x,y):
    fx = f(x)
    fy = f(y)
    result = np.inner(fx, fy)
    return result

def k(x,y):
    result = np.inner(x, y)**2
    return result


if __name__=="__main__":
    x = np.array([1,2,3]).reshape(1,3)
    y = np.array([4,5,6]).reshape(1,3)

    print("原始输入维数:", x.shape[1])
    print("映射空间维数:", f(x).shape[1])

    n = 100000
    start = time.time()
    for i in range(n):
        r1 = fx_fy(x, y)
    end1 = time.time()
    for i in range(n):
        r2 = k(x, y)
    end2 = time.time()
    print(r1,r2)
    print("fx_fy(x,y)运行时间:", end1-start)
    print("K(x,y)运行时间:", end2-end1)
