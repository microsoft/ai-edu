
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.svm import SVC
from sklearn.preprocessing import *
import sys
import os
from pathlib import Path
import matplotlib as mpl


def generate_file_path(file_name):
    curr_path = sys.argv[0]
    curr_dir = os.path.dirname(curr_path)
    file_path = os.path.join(curr_dir, file_name)
    return file_path

def load_data(file_name):
    file_path = generate_file_path(file_name)
    file = Path(file_path)
    if file.exists():
        samples = np.loadtxt(file, delimiter=',')
        X = samples[:, 0:2]
        Y = samples[:, 2]
    else:
        X, Y = make_circles(n_samples=100, factor=0.5, noise=0.1)
        Y[Y == 0] = -1
        samples = np.hstack((X,Y.reshape(-1,1)))
        np.savetxt(file_path, samples, fmt='%f, %f, %d', delimiter=',', header='x1, x2, y')
    return X, Y

def draw_circle_3d(ax, x, y):
    ax.scatter(x[y==1,0], x[y==1,1], x[y==1,2], marker='^', c='r')
    ax.scatter(x[y==-1,0], x[y==-1,1], x[y==-1,2], marker='o', c='b')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")

def draw_2d(ax, x, y):
    ax.scatter(x[y==1,0], x[y==1,1], marker='^', c='r')
    ax.scatter(x[y==-1,0], x[y==-1,1], marker='o', c='b')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

if __name__=="__main__":

    X_raw, Y = load_data("Data_12_circle_100.csv")
   
    fig = plt.figure()
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    
    # 原始样本
    ax1 = fig.add_subplot(131)
    ax1.axis('equal')
    ax1.grid()
    ax1.set_title(u"原始样本数据")
    draw_2d(ax1, X_raw, Y)
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    
    # 重新构建二维特征
    ax2 = fig.add_subplot(132)
    ax2.axis('equal')
    ax2.grid()
    ax2.set_title(u"重新构建二维特征")
    
    # 前两维数据都变成自身的平方，不增加新维
    X_2d = np.zeros_like(X_raw)
    X_2d[:,0] = X_raw[:,0]**2
    X_2d[:,1] = X_raw[:,1]**2
    draw_2d(ax2, X_2d, Y)
  
    # 构建三维特征
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title(u"构建三维特征")
    
    # 前两维特征不变
    X_3d = np.zeros((X_raw.shape[0], 3))
    X_3d[:,0:2] = X_raw
    # 增加一维
    X_3d[:,2] = X_raw[:,0]**2 + X_raw[:,1]**2
    '''
    X_3d[:,0] = X_raw[:,0]**2
    X_3d[:,1] = X_raw[:,0]*X_raw[:,1]
    X_3d[:,2] = X_raw[:,1]**2
    '''
    draw_circle_3d(ax3, X_3d, Y)

    plt.show()
