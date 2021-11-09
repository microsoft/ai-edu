
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.svm import *
from sklearn.pipeline import *
from sklearn.preprocessing import *
import sys
import os
from pathlib import Path
import matplotlib as mpl
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


# 显示各个样本独立的二维高斯函数平面投影
def show_sample_gaussian(ax, gamma, X, Y, scope):
    set_ax(ax, scope)
    ax.set_title(u"以各个样本为中心的高斯函数投影")

    for idx in range(X.shape[0]):
        P,Q,R = gaussian_2d(X[idx], gamma, Y[idx], scope)
        if (Y[idx]==1):
            c = 'red'
        else:
            c = 'blue'
        ax.contour(P,Q,R,2,colors=c,linewidths=[0.5,0.2],linestyles='dashed')

    draw_2d_samples(ax, X, Y)


# 绘图区基本设置
def set_ax(ax, scope):
    ax.axis('equal')
    ax.grid()
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

 

# 显示分类区域结果
def show_predication_result(ax, model, X, Y, scope):   
    set_ax(ax, scope)

    # 生成测试数据，形成一个点阵来模拟平面
    x1 = np.linspace(scope[0], scope[1], scope[2])
    x2 = np.linspace(scope[3], scope[4], scope[5])
    X1,X2 = np.meshgrid(x1,x2)
    # 从行列变形为序列数据
    X12 = np.c_[X1.ravel(), X2.ravel()]
    # 做预测
    pred = model.predict(X12)
    # 从序列数据变形行列形式
    y_pred = pred.reshape(X1.shape)

    # 绘图
    cmap = ListedColormap(['yellow','lightgray'])
    plt.contourf(X1,X2, y_pred, cmap=cmap)
    # 绘制原始样本点用于比对
    draw_2d_samples(ax, X, Y)

# 高斯核函数 SVM
def rbf_svc(X, Y, C, gamma):
    model = SVC(C=C, gamma = gamma, kernel='rbf')
    model.fit(X,Y)

    print("权重:",np.dot(model.dual_coef_, model.support_vectors_))
    print("支持向量个数:",model.n_support_)
    print("支持向量索引:",model.support_)
    print("支持向量:",np.round(model.support_vectors_,3))
    print("支持向量ay:",np.round(model.dual_coef_,3))
    print("准确率:", model.score(X, Y))

    return model


if __name__=="__main__":

    file_name = "Data_12_moon_10.csv"
    X_raw, Y = load_data(file_name, 10)

    ss = StandardScaler()
    X = ss.fit_transform(X_raw)
    #print("X 标准化后的值：")
    #print(X)

    # 基本绘图设置
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.axis('off')
    
    # 用rbf SVM 做分类    
    gamma = 2
    C = 2
    model = rbf_svc(X, Y, C, gamma)

    # 显示高斯函数投影
    scope = [-3,3,100,-3,3,100]
    ax1 = fig.add_subplot(121)
    show_sample_gaussian(ax1, gamma, X, Y, scope)

    # 显示分类预测结果
    ax2 = fig.add_subplot(122)
    ax2.set_title(u"区域分类结果")
    show_predication_result(ax2, model, X, Y, scope)

    plt.show()


    fig = plt.figure()
    plt.axis('off')

    gamma = 4
    C = 2
    model = rbf_svc(X, Y, C, gamma)
    ax1 = fig.add_subplot(131)
    show_predication_result(ax1, model, X, Y, scope)
    ax1.set_title(u"gamma=4")

    gamma = 1
    C = 2
    model = rbf_svc(X, Y, C, gamma)
    ax2 = fig.add_subplot(132)
    show_predication_result(ax2, model, X, Y, scope)
    ax2.set_title(u"gamma=1")


    gamma = 0.5
    C = 2
    model = rbf_svc(X, Y, C, gamma)
    ax3 = fig.add_subplot(133)
    show_predication_result(ax3, model, X, Y, scope)
    ax3.set_title(u"gamma=0.5")

    plt.show()
