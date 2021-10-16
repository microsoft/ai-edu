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

# 公式 1.6.2
def least_square_2(X,Y):
    n = X.shape[0]
    # a_hat
    numerator = np.cov(X, Y, rowvar=False, bias=False)[0,1]
    denominator = np.var(X)
    a_hat = numerator / denominator
    # b_hat
    b_hat = (np.sum(Y - a_hat * X))/n
    return a_hat, b_hat

# 公式1.6.3, 1.6.4
def least_square_3(X,Y):
    n = X.shape[0]
    # a_hat 公式1.6.3
    numerator = np.sum((X-np.mean(X))*(Y-np.mean(Y)))
    denominator = np.sum((X-np.mean(X))*(X-np.mean(X)))
    a_hat = numerator / denominator
    # b_hat 公式1.6.4
    b_hat = np.mean(Y) - a_hat * np.mean(X)
    return a_hat, b_hat


if __name__ == '__main__':
    file_name = "1-0-data.csv"
    samples = load_file(file_name)
    if samples is not None:
        X = samples[:, 0].reshape(200,1)
        Y = samples[:, 1].reshape(200,1)
        a_hat, b_hat = least_square_2(X,Y)
        print(str.format("a_hat={0:.4f}, b_hat={1:.4f}",a_hat,b_hat))
        a_hat, b_hat = least_square_3(X,Y)
        print(str.format("a_hat={0:.4f}, b_hat={1:.4f}",a_hat,b_hat))
    else:
        print("cannot find file " + file_name)
