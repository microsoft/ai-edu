import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

def load_file(file_name):
    file_path = generate_file_path(file_name)
    file = Path(file_path)
    if file.exists():
        samples = np.loadtxt(file, delimiter=',')
        return samples
    else:
        return None

def generate_file_path(file_name):
    curr_path = sys.argv[0]
    curr_dir = os.path.dirname(curr_path)
    file_path = os.path.join(curr_dir, file_name)
    return file_path

def normal_equation(X,Y):
    num_example = X.shape[0]
    # 在原始的X矩阵最左侧加一列1
    ones = np.ones((num_example,1))
    x = np.column_stack((ones, X))    
    # X^T * X
    p = np.dot(x.T, x)
    # (X^T * X)^{-1}
    q = np.linalg.inv(p)
    # (X^T * X)^{-1} * X^T
    r = np.dot(q, x.T)
    # (X^T * X)^{-1} * X^T * Y
    A = np.dot(r, Y)
    # 按顺序
    b = A[0,0]
    a1 = A[1,0]
    a2 = A[2,0]
    return a1, a2, b

if __name__ == '__main__':
    file_name = "2-0-data.csv"
    samples = load_file(file_name)
    if (samples is not None):
        X = samples[:,0:2]
        Y = samples[:,-1].reshape(-1,1)
        a1, a2, b = normal_equation(X,Y)
        print(str.format("a1={0:.4f}, a2={1:.4f}, b={2:.4f}", a1, a2, b))
    else:
        print("cannot find data file:" + file_name)
