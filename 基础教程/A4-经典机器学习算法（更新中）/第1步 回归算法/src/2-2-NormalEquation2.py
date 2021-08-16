import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl


def generate_file_path(file_name):
    curr_path = sys.argv[0]
    curr_dir = os.path.dirname(curr_path)
    file_path = os.path.join(curr_dir, file_name)
    return file_path

def load_file(file_name):
    file_path = generate_file_path(file_name)
    file = Path(file_path)
    if file.exists():
        samples = np.loadtxt(file, delimiter=',')
        return samples
    else:
        return None

def normal_equation(X,Y):
    num_example = X.shape[0]
    # 在原始的X矩阵最左侧加一列1
    ones = np.ones((num_example,1))
    x = np.column_stack((ones, X))    
    # X^T * X
    a = np.dot(x.T, x)
    # (X^T * X)^{-1}
    b = np.linalg.inv(a)
    # (X^T * X)^{-1} * X^T
    c = np.dot(b, x.T)
    # (X^T * X)^{-1} * X^T * Y
    d = np.dot(c, Y)
    # 按顺序
    b = d[0,0]
    a = d[1,0]
    return a, b

if __name__ == '__main__':
    file_name = "1-0-data.csv"
    samples = load_file(file_name)
    if samples is not None:
        X = samples[:, 0].reshape(200,1)
        Y = samples[:, 1].reshape(200,1)
        a, b = normal_equation(X,Y)
        print(str.format("a={0:.4f}, b={1:.4f}", a,b))
    else:
        print("cannot find file " + file_name)
