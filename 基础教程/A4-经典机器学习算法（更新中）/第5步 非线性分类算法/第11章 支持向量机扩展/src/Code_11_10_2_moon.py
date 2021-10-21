
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
from Code_11_9_2_Xor import *

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

# 绘图区基本设置
def set_ax(ax, scope):
    if (ax.name != '3d'):
        ax.grid()
        ax.axis('equal')
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    if (scope is not None):
        ax.set_xlim(scope[0], scope[1])
        ax.set_ylim(scope[3], scope[4])

# 绘制平面样本点
def draw_2d_samples(ax, x, y, display_text=True):
    ax.scatter(x[y==1,0], x[y==1,1], marker='^', color='red')
    ax.scatter(x[y==-1,0], x[y==-1,1], marker='o', color='blue')
    if (display_text):
        for i in range(x.shape[0]):
            ax.text(x[i,0], x[i,1]+0.1, str(i))

# 高斯核函数图像
def gaussian_kernal(gamma, landmark, weight, scope):
    xx = np.linspace(scope[0], scope[1], scope[2])
    yy = np.linspace(scope[3], scope[4], scope[5])
    P,Q = np.meshgrid(xx, yy)
    R = weight * np.exp(-gamma * ((P-landmark[0])**2 + (Q-landmark[1])**2))
    return P,Q,R
  

# 显示独立的二维高斯函数，和平面投影
def show_result_1(gamma, model, X, Y, scope):
    # 基本绘图设置
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.axis('off')
    plt.title(u"简化月亮数据集 以独立样本为中心的高斯函数示意图")

    ax1 = fig.add_subplot(121, projection='3d')
    set_ax(ax1, scope)

    ax2 = fig.add_subplot(122)
    set_ax(ax2, scope)

    R_pos = None
    R_neg = None
    for idx in model.support_:
        #P,Q,R = gaussian_3d(gamma, X[idx], scope)
        P,Q,R = gaussian_kernal(gamma, X[idx], Y[idx], scope)
        if (Y[idx]==1):
            c = 'red'
        else:
            c = 'blue'
        ax2.contour(P,Q,R,2,zdir='z',offset=0,colors=c,linewidths=[0.5,0.2],linestyles='dashed')
        
        if (Y[idx]==1):
            if (R_pos is None):
                R_pos = R
            else:
                # 取maximum是不想有叠加效果，即保证两个样本的高斯函数的独立
                R_pos = np.maximum(R_pos, R)
        else:   # -1
            if (R_neg is None):
                R_neg  = R
            else:
                R_neg = np.minimum(R_neg, R)
    

    #ax1.plot_surface(P, Q, R_pos - R_neg, cmap=cm.coolwarm)
    ax1.plot_surface(P, Q, R_pos, cmap=cm.coolwarm)
    ax1.plot_surface(P, Q, R_neg, cmap=cm.coolwarm)

    for i in range(X.shape[0]):
        if(Y[i]==1):
            ax1.text(X[i,0], X[i,1], 1, str(i))
        else:
            ax1.text(X[i,0], X[i,1], -1.1, str(i))


    draw_2d_samples(ax2, X, Y)

    plt.show()

# 显示高斯核函数，即所有样本点之间的内积
def show_result_2(gamma, X, weights, scope, title):
    # 基本绘图设置
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.title(title)
    plt.axis('off')

    RR = None
    for i in range(X.shape[0]):
        P,Q,R = gaussian_kernal(gamma, X[i], weights[i], scope)
        if (RR is None):
            RR = R
        else:
            # 高斯核函数的各个样本点之间有叠加关系
            RR += R

    ax1 = fig.add_subplot(121, projection='3d')
    set_ax(ax1, scope)
    ax1.plot_surface(P, Q, RR, cmap=cm.coolwarm)

    ax2 = fig.add_subplot(122)
    set_ax(ax2, scope)
    ax2.contour(P,Q,RR,zdir='z',offset=0,cmap=cm.coolwarm)                     #绘制xy面投影
    draw_2d_samples(ax2, X, Y)

    plt.show()


if __name__=="__main__":

    file_name = "Data_moon_10.csv"
    X_raw, Y = load_data(file_name, 10)

    ss = StandardScaler()
    X = ss.fit_transform(X_raw)
    print("X 标准化后的值：")
    print(X)

    gamma = 2
    X_new = K_matrix(X, X, gamma)
    print("映射结果：")
    print(np.round(X_new,3))

    # 尝试用线性 SVM 做分类    
    C = 2
    model = linear_svc(X_new, Y, C)
    # 显示分类预测结果
    scope = [-3,3,100,-3,3,100]
    X1, X2, y_pred = prediction(model, gamma, X, scope)
    
    show_result_1(gamma, model, X, Y, scope)
    title = u"所有样本的高斯核函数示意图"
    show_result_2(gamma, X, Y, scope, title)
    title = u"所有样本（含权重）的高斯核函数示意图"
    show_result_2(gamma, X, model.dual_coef_[0], scope, title)
