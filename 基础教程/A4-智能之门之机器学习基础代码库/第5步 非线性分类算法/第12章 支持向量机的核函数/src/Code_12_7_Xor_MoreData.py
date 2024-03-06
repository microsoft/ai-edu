
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.preprocessing import *
from sklearn.svm import *
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

def linear_svc(X,Y,C):
    model = SVC(C=C, kernel='linear')
    model.fit(X,Y)

    #print("权重:", np.round(model.coef_, 3))
    #print("权重5x5:\n", np.round(model.coef_.reshape(5,5),2))
    #print("支持向量个数:",model.n_support_)
    #print("支持向量索引:",model.support_)
    #print("支持向量:",np.round(model.support_vectors_,3))
    #print("支持向量 a*y:", model.dual_coef_)
    score = model.score(X, Y)
    print("准确率:", score)

    return model, score
    

# 生成测试数据，形成一个点阵来模拟平面
def prediction(model, gamma, landmark, scope):
    # 生成测试数据，形成一个点阵来模拟平面
    x1 = np.linspace(scope[0], scope[1], 50)
    x2 = np.linspace(scope[3], scope[4], 50)
    X1,X2 = np.meshgrid(x1,x2)
    X12 = np.c_[X1.ravel(), X2.ravel()]
    # 用与生成训练数据相同的函数来生成测试数据特征
    X12_new = Feature_matrix(X12, landmark, gamma)
    # 做预测
    pred = model.predict(X12_new)
    # 变形并绘制分类区域
    y_pred = pred.reshape(X1.shape)
    #prob = model.decision_function(X12_new)

    return X1, X2, y_pred

# 展示分类结果
def show_result(ax, X1, X2, y_pred, X, Y):
    cmap = ListedColormap(['yellow','lightgray'])
    ax.contourf(X1,X2, y_pred, cmap=cmap)
    draw_2d(ax, X, Y, False)

def draw_2d(ax, x, y, display_text=True):
    ax.scatter(x[y==1,0], x[y==1,1], marker='^', color='red')
    ax.scatter(x[y==-1,0], x[y==-1,1], marker='o', color='blue')
    if (display_text):
        for i in range(x.shape[0]):
            ax.text(x[i,0], x[i,1]+0.1, str(i))

# 映射特征矩阵
# X - 样本数据
# L - 地标 Landmark，在此例中就是样本数据
# gamma - 形状参数
def Feature_matrix(X, L, gamma):
    n = X.shape[0]  # 样本数量
    m = L.shape[0]  # 特征数量
    X_feature = np.zeros(shape=(n,m))
    for i in range(n):
        for j in range(m):
            # 计算每个样本点在地标上的高斯函数值，式 11.10.1 
            X_feature[i,j] = np.exp(-gamma * np.linalg.norm(X[i] - L[j])**2)

    return X_feature

def create_landmark(scope):
    x1 = np.linspace(scope[0], scope[1], scope[2])
    # 从1到-0.5，相当于y值从上向下数，便于和图像吻合
    x2 = np.linspace(scope[4], scope[3], scope[5])

    landmark = np.zeros((scope[2]*scope[5], 2))
    for i in range(scope[2]):
        for j in range(scope[5]):
            landmark[i*scope[2]+j,0] = x1[j]
            landmark[i*scope[2]+j,1] = x2[i]

    return landmark

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

    
import sys
import os
from pathlib import Path

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
        X, Y = make_xor(n_samples=100)
        samples = np.hstack((X,Y.reshape(-1,1)))
        np.savetxt(file_path, samples, fmt='%f, %f, %d', delimiter=',', header='x1, x2, y')
    return X, Y

def make_xor(n_samples=100):
    X = np.random.normal(0, 1, (120,2))
    X_xor = []
    for x in X:
        if (abs(x[0]) > 0.2 and abs(x[1]) > 0.2):
            X_xor.append(x)
    X = np.array(X_xor)
    y_xor = np.logical_xor(X[:, 0] > 0 , X[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)
    return X, y_xor    


def classify(ax, scope):
    # 创建地标
    landmark = create_landmark(scope)

    # 特征映射
    gamma = 1
    X_feature = Feature_matrix(X, landmark, gamma)

    # 线性分类
    C = 1
    model, score = linear_svc(X_feature, Y, C)

    # 可视化结果
    X1,X2,y_pred = prediction(model, gamma, landmark, scope)
    show_result(ax, X1, X2, y_pred, X, Y)
    ax.set_title(str.format("地标密度 {0}x{1}, 准确率 {2:.2f}", scope[2], scope[5], score))


if __name__=="__main__":
    X, Y = load_data("Data_12_xor.csv")
    
    # 基本绘图设置
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()

    # 5x5=25
    scope = [-3,3,5, -3,3,5]
    ax1 = fig.add_subplot(131)
    set_ax(ax1, scope)
    classify(ax1, scope)

    # 10x10=100
    scope = [-3,3,10, -3,3,10]
    ax2 = fig.add_subplot(132)
    set_ax(ax2, scope)
    classify(ax2, scope)

    # 20x20=400
    scope = [-3,3,20, -3,3,20]
    ax3 = fig.add_subplot(133)
    set_ax(ax3, scope)
    classify(ax3, scope)

    plt.show()
