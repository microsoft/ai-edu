
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from numpy.linalg.linalg import norm
from sklearn.datasets import *
from sklearn.svm import *
from sklearn.pipeline import *
from sklearn.preprocessing import *
import sys
import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.cm as cm
from Code_5_11_3_xor import *

def generate_file_path(file_name):
    curr_path = sys.argv[0]
    curr_dir = os.path.dirname(curr_path)
    file_path = os.path.join(curr_dir, file_name)
    return file_path

def load_data(file_name, n_samples):
    file_path = generate_file_path(file_name)
    file = Path(file_path)
    if file.exists():
        samples = np.loadtxt(file, delimiter=',')
        X = samples[:, 0:2]
        Y = samples[:, 2]
    else:
        X, Y = make_moons(n_samples=n_samples, noise=0.1, shuffle=False)
        Y[Y == 0] = -1
        samples = np.hstack((X,Y.reshape(-1,1)))
        np.savetxt(file_path, samples, fmt='%f, %f, %d', delimiter=',', header='x1, x2, y')
    return X, Y

def set_ax(ax, scope):
    if (ax.name != '3d'):
        ax.grid()
        ax.axis('equal')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    if (scope is not None):
        ax.set_xlim(scope[0], scope[1])
        ax.set_ylim(scope[3], scope[4])

def show_samples(X_10, Y_10, X_100, Y_100, scope):
    # 基本绘图设置
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.axis('off')
    plt.title(u"月亮数据集的100个样本（左）和10个样本（右）")

    ax1 = fig.add_subplot(121)
    set_ax(ax1, scope)
    draw_2d(ax1, X_100, Y_100, False)

    ax2 = fig.add_subplot(122)
    set_ax(ax2, scope)
    draw_2d(ax2, X_10, Y_10, True)

    plt.show()

def draw_2d(ax, x, y, display_text=True):
    ax.scatter(x[y==1,0], x[y==1,1], marker='^', color='red')
    ax.scatter(x[y==-1,0], x[y==-1,1], marker='o', color='blue')
    if (display_text):
        for i in range(x.shape[0]):
            ax.text(x[i,0], x[i,1]+0.1, str(i))

# 三维高斯图像
def gaussian_kernal(gamma, landmark, weight, scope):
    xx = np.linspace(scope[0], scope[1], scope[2])
    yy = np.linspace(scope[3], scope[4], scope[5])
    P,Q = np.meshgrid(xx, yy)
    R = weight * np.exp(-gamma * ((P-landmark[0])**2 + (Q-landmark[1])**2))
    return P,Q,R

def gaussian_3d(gamma, center, scope):
    xx = np.linspace(scope[0], scope[1], scope[2])
    yy = np.linspace(scope[3], scope[4], scope[5])
    P,Q = np.meshgrid(xx, yy)
    #ax.plot_surface(P, Q, R1, cmap=cm.coolwarm)
    R = np.exp(-gamma * ((P-center[0])**2/2 + (Q-center[1])**2/2))
    return P, Q, R
   

# 显示 1 标准化后的样本 以及 2 分类结果
def show_result_1(X1, X2, y_pred, X, Y, scope):

    # 基本绘图设置
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.title(u"简化月亮数据集及分类结果")
    plt.grid()
    plt.axis('off')

    # 绘图
    ax1 = fig.add_subplot(121)
    set_ax(ax1, scope)
    draw_2d(ax1, X, Y)

    ax2 = fig.add_subplot(122)
    set_ax(ax2, scope)
    ax2.contourf(X1,X2, y_pred)
    # 绘制原始样本点用于比对
    draw_2d(ax2, X, Y)

    plt.show()


# 显示独立的二维高斯函数，和平面投影
def show_result_2(model, X, Y, scope):
    # 基本绘图设置
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.title(u"简化月亮数据集 以独立样本为中心的高斯函数示意图")
    plt.grid()
    plt.axis('off')

    ax1 = fig.add_subplot(121)
    set_ax(ax1, scope)
    ax2 = fig.add_subplot(122, projection='3d')
    set_ax(ax2, scope)

    R_pos = None
    R_neg = None
    for idx in model.support_:
        P,Q,R = gaussian_3d(gamma, X[idx], scope)
        if (Y[idx]==1):
            c = 'red'
        else:
            c = 'blue'
        ax1.contour(P,Q,R,2,zdir='z',offset=0,colors=c,linewidths=[0.5,0.2],linestyles='dashed')
        
        if (Y[idx]==1):
            if (R_pos is None):
                R_pos = R
            else:
                R_pos = np.maximum(R_pos, R)
        else:   # -1
            if (R_neg is None):
                R_neg  = R
            else:
                R_neg = np.maximum(R_neg, R)
    

    draw_2d(ax1, X, Y)

    ax2.plot_surface(P, Q, R_pos - R_neg, cmap=cm.coolwarm)

    plt.show()


def show_result_3(model, X, Y, scope):
    # 基本绘图设置
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.title(u"简化月亮数据集 以所有样本为地标的高斯核函数示意图")
    plt.grid()
    plt.axis('off')

    RR = None
    for i in range(np.sum(model.n_support_)):
        P,Q,R = gaussian_kernal(gamma, X[i], model.dual_coef_[0,i], scope)
        if (RR is None):
            RR = R
        else:
            RR += R

    ax1 = fig.add_subplot(121)
    set_ax(ax1, scope)

    ax1.contour(P,Q,RR,zdir='z',offset=0,cmap=cm.coolwarm)                     #绘制xy面投影
    draw_2d(ax1, X, Y)

    ax2 = fig.add_subplot(122, projection='3d')
    set_ax(ax2, scope)
    ax2.plot_surface(P, Q, RR, cmap=cm.coolwarm)

    plt.show()


if __name__=="__main__":

    file_name = "Data_moon_10.csv"
    X_10, Y_10 = load_data(file_name, 10)

    file_name = "Data_moon_100.csv"
    X_100, Y_100 = load_data(file_name, 100)

    show_samples(X_10, Y_10, X_100, Y_100, None)

    ss = StandardScaler()
    X = ss.fit_transform(X_10)
    print("X 标准化后的值：")
    print(X)

    gamma = 2
    X_new = K_matrix(X, X, gamma)
    print("映射结果：")
    print(np.round(X_new,3))

    # 尝试用线性 SVM 做分类    
    C = 2
    model = linear_svc(X_new, Y_10, C)
    # 显示分类预测结果
    scope = [-2,2,100,-2,2,100]
    X1, X2, y_pred = prediction(model, gamma, X, scope)
    
    show_result_1(X1, X2, y_pred, X, Y_10, scope)
    show_result_2(model, X, Y_10, scope)
    show_result_3(model, X, Y_10, scope)
