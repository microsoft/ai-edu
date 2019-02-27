import numpy as np


if __name__ == '__main__':
    x=np.array([1,2104,5,1,45,1,1416,3,2,40,1,1534,3,2,40]).reshape(3,5)
    print(x[:,1:5])
    y=np.array([460,232,315]).reshape(3,1)
    b=np.dot(x.T, x)
    print(b)
    c=np.asmatrix(b)
    print(c)
    d=np.linalg.inv(c)
    print(d)
    d1=np.asarray(d)
    e=np.dot(d1,x.T)
    f=np.dot(e,y)

    print(f)
