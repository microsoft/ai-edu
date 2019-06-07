import numpy as np
from pathlib import Path

from SimpleDataReader import *

file_name = "../../data/ch05.npz"

def LoadData():
    reader = SimpleDataReader(file_name)
    reader.ReadData()
    X,Y = reader.GetWholeTrainSamples()
    return X, Y

if __name__ == '__main__':
    X,Y = LoadData()
    num_example = X.shape[0]
    one = np.ones((num_example,1))
    x = np.column_stack((one, (X[0:num_example,:])))

    a = np.dot(x.T, x)
    # need to convert to matrix, because np.linalg.inv only works on matrix instead of array
    b = np.asmatrix(a)
    c = np.linalg.inv(b)
    d = np.dot(c, x.T)
    e = np.dot(d, Y)
    #print(e)
    b=e[0,0]
    w1=e[1,0]
    w2=e[2,0]
    w3=e[3,0]
    print("w1=", w1)
    print("w2=", w2)
    print("w3=", w3)
    print("b=", b)
    # inference
    z = w1 * 2 + w2 * 5 + w3 * 93 + b
    print("z=",z)
