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
    a1=e[0,1]
    a2=e[0,2]
    return a1, a2, b

if __name__ == '__main__':
    file_name = "2-0-data.csv"
    samples = load_file(file_name)
    if (samples is not None):
        X = samples[:,0:2]
        Y = samples[:,-1]
        a1, a2, b = normal_equation(X,Y)
        print(str.format("a1={0:.4f}, a2={1:.4f}, b={2:.4f}", a1, a2, b))
    else:
        print("cannot find data file:" + file_name)
