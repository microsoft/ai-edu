
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
from matplotlib.colors import ListedColormap

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

# 高斯函数图像
# sample - 样本点
# gamma - 控制形状
# weight - 控制权重（正负及整体升高或降低）
# scope - 网格数据范围及密度
def gaussian_2d(sample, gamma, weight, scope):
    # 生成网格数据
    xx = np.linspace(scope[0], scope[1], scope[2])
    yy = np.linspace(scope[3], scope[4], scope[5])
    P,Q = np.meshgrid(xx, yy)
    # 二维高斯函数 * 权重 -> 三维高斯曲面
    R = weight * np.exp(-gamma * ((P-sample[0])**2 + (Q-sample[1])**2))
    return P,Q,R

# 显示各个样本独立的二维高斯函数和其平面投影
def show_sample_gaussian(gamma, X, Y, scope):
    # 基本绘图设置
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.axis('off')
    plt.title(u"以各个样本为中心的独立高斯函数示意图")

    ax1 = fig.add_subplot(121, projection='3d')
    set_ax(ax1, scope)

    ax2 = fig.add_subplot(122)
    set_ax(ax2, scope)

    R_pos = None
    R_neg = None
    for idx in range(X.shape[0]):
        P,Q,R = gaussian_2d(X[idx], gamma, Y[idx], scope)
        if (Y[idx]==1):
            c = 'red'
        else:
            c = 'blue'
        ax2.contour(P,Q,R,2,colors=c,linewidths=[0.5,0.2],linestyles='dashed')
        
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
    
    # 绘制所有正类样本的高斯曲面
    ax1.plot_surface(P, Q, R_pos, cmap=cm.coolwarm)
    # 绘制所有负类样本的高斯曲面
    ax1.plot_surface(P, Q, R_neg, cmap=cm.coolwarm)
    # 显示样本序号
    for i in range(X.shape[0]):
        if(Y[i]==1):
            ax1.text(X[i,0], X[i,1], 1, str(i))
        else:
            ax1.text(X[i,0], X[i,1], -1.1, str(i))

    draw_2d_samples(ax2, X, Y)

    plt.show()

# 显示高斯核函数，即所有样本点之间的内积
def show_result_2(model, gamma, X, scope, title, X1, X2, y_pred):
    # 基本绘图设置
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.title(title)
    plt.axis('off')

    RR = None
    for i in range(len(model.support_)):
        sv_id = model.support_[i]
        #P,Q,R = gaussian_2d(X[sv_id], gamma, model.dual_coef_[0, i], scope)
        P,Q,R = gaussian_2d(X[sv_id], gamma, Y[sv_id], scope)
        if (RR is None):
            RR = R
        else:
            # 高斯核函数的各个样本点之间有叠加关系
            RR += R

    ax1 = fig.add_subplot(131, projection='3d')
    set_ax(ax1, scope)
    ax1.plot_surface(P, Q, RR, cmap=cm.coolwarm)

    ax2 = fig.add_subplot(132)
    set_ax(ax2, scope)
    ax2.contour(P, Q, RR, cmap=cm.coolwarm)
    draw_2d_samples(ax2, X, Y)

    ax3 = fig.add_subplot(133)
    set_ax(ax3, scope)
    cmap = ListedColormap(['yellow','lightgray'])
    ax3.contourf(X1,X2, y_pred, cmap=cmap)
    # 绘制原始样本点用于比对
    draw_2d_samples(ax3, X, Y)


    plt.show()

# 生成网格点作为landmark
def create_landmark(X, scope, gamma):
    x1 = np.linspace(scope[0], scope[1], scope[2])
    x2 = np.linspace(scope[3], scope[4], scope[5])

    landmark = np.zeros((scope[2]*scope[5], 2))
    for i in range(scope[2]):
        for j in range(scope[5]):
            landmark[i*scope[2]+j,0] = x1[i]
            landmark[i*scope[2]+j,1] = x2[j]

    return landmark

def linear_svc(X,Y,C):
    model = SVC(C=C, kernel='linear')
    model.fit(X,Y)

    print("权重:",model.coef_)
    print("支持向量个数:",model.n_support_)
    print("支持向量索引:",model.support_)
    #print("支持向量:",np.round(model.support_vectors_,3))
    print("支持向量ay:",model.dual_coef_)
    print("准确率:", model.score(X, Y))

    return model

def Feature_matrix(X, L, gamma):
    n = X.shape[0]  # 样本数量
    m = L.shape[0]  # 特征数量
    Features = np.zeros(shape=(n,m))
    for i in range(n):
        for j in range(m):
            # 计算每个样本点在网格标记点上的高斯函数值
            Features[i,j] = np.exp(-gamma * np.linalg.norm(X[i] - L[j])**2)

    return Features

# 生成测试数据，形成一个点阵来模拟平面
def prediction(model, gamma, landmark, scope):
    # 生成测试数据，形成一个点阵来模拟平面
    x1 = np.linspace(scope[0], scope[1], scope[2])
    x2 = np.linspace(scope[3], scope[4], scope[5])
    X1,X2 = np.meshgrid(x1,x2)
    # 从行列变形为序列数据
    X12 = np.c_[X1.ravel(), X2.ravel()]
    # 用与生成训练数据相同的函数来生成测试数据特征
    X12_new = Feature_matrix(X12, landmark, gamma)
    # 做预测
    pred = model.predict(X12_new)
    # 从序列数据变形行列形式
    y_pred = pred.reshape(X1.shape)

    return X1, X2, y_pred

if __name__=="__main__":

    file_name = "Data_moon_10.csv"
    X_raw, Y = load_data(file_name, 10)

    ss = StandardScaler()
    X = ss.fit_transform(X_raw)
    print("X 标准化后的值：")
    print(X)

    gamma = 2
    scope = [-3,3,13,-3,3,13]
    landmark = create_landmark(X, scope, gamma)
    X_new = Feature_matrix(X, landmark, gamma)


    # 尝试用线性 SVM 做分类    
    C = 1
    model = linear_svc(X_new, Y, C)
    # 显示分类预测结果
    scope = [-3,3,9,-3,3,9]
    # 显示各个样本独立的二维高斯函数和其平面投影
    show_sample_gaussian(gamma, X, Y, scope)

    X1, X2, y_pred = prediction(model, gamma, landmark, scope)

    xxxx = Feature_matrix(landmark, landmark, gamma)
    a = model.decision_function(xxxx)
    b = a.reshape(13,13)
   
    xx = np.linspace(scope[0], scope[1], 13)
    yy = np.linspace(scope[3], scope[4], 13)
    P,Q = np.meshgrid(xx, yy)
    fig = plt.figure()
    plt.grid()
    plt.axis('equal')
    plt.contour(P, Q, b, cmap=cm.coolwarm)
    plt.show()


    title = u"所有样本（含权重）的高斯核函数示意图"
    show_result_2(model, gamma, X, scope, title, X1, X2, y_pred)
