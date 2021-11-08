
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

    print("权重:", np.round(model.coef_, 3))
    print("权重5x5:\n", np.round(model.coef_, 3).reshape(5,5))
    print("支持向量个数:",model.n_support_)
    print("支持向量索引:",model.support_)
    print("支持向量:",np.round(model.support_vectors_,3))
    print("支持向量 a*y:", model.dual_coef_)
    print("准确率:", model.score(X, Y))

    return model
    

# 生成测试数据，形成一个点阵来模拟平面
def prediction(model, gamma, landmark, scope):
    # 生成测试数据，形成一个点阵来模拟平面
    x1 = np.linspace(scope[0], scope[1], scope[2])
    x2 = np.linspace(scope[3], scope[4], scope[5])
    X1,X2 = np.meshgrid(x1,x2)
    X12 = np.c_[X1.ravel(), X2.ravel()]
    # 用与生成训练数据相同的函数来生成测试数据特征
    X12_new = Feature_matrix(X12, landmark, gamma)
    # 做预测
    pred = model.predict(X12_new)
    # 变形并绘制分类区域
    y_pred = pred.reshape(X1.shape)
    prob = model.decision_function(X12_new)

    return X1, X2, y_pred, prob

# 展示分类结果
def show_result(X1, X2, y_pred, X_sample, Y, prob, scope3):
    # 基本绘图设置
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.title(u"异或问题的分类结果")
    plt.axis('off')

    ax1 = fig.add_subplot(121)
    set_ax(ax1, scope3)
    # 绘图
    cmap = ListedColormap(['yellow','lightgray'])
    ax1.contourf(X1,X2, y_pred, cmap=cmap)
    # 绘制原始样本点用于比对
    draw_2d(ax1, X_sample, Y)

    ax2 = fig.add_subplot(122)
    set_ax(ax2, scope3)
    R = prob.reshape(scope3[2],scope3[5])
    xx = np.linspace(scope3[0], scope3[1], scope3[2])
    yy = np.linspace(scope3[3], scope3[4], scope3[5])
    P,Q = np.meshgrid(xx, yy)
    ax2.contour(P, Q, R, levels=np.linspace(-1.5, 1.5, 20), cmap=cm.coolwarm)
    
    plt.show()

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
    #scope = [-0.5,1.5,5, -0.5,1.5,5]
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

# 高斯函数图像
# sample - 样本点
# gamma - 控制形状
# weight - 控制权重（正负及整体升高或降低）
# scope - 网格数据范围及密度
def gaussian_2d_fun(sample, gamma, weight, scope):
    # 生成网格数据
    xx = np.linspace(scope[0], scope[1], scope[2])
    yy = np.linspace(scope[3], scope[4], scope[5])
    P,Q = np.meshgrid(xx, yy)
    # 二维高斯函数 * 权重 -> 三维高斯曲面
    R = weight * np.exp(-gamma * ((P-sample[0])**2 + (Q-sample[1])**2))
    return P,Q,R

def show_gaussian_2d(gamma, X, Y, scope, L):
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title(u'样本的二维高斯曲面')
    set_ax(ax1, scope)
    ax2 = fig.add_subplot(122)
    ax2.set_title(u'样本的二维高斯曲面投影')
    set_ax(ax2, scope)

    RR = None
    for i in range(X.shape[0]):
        P,Q,R = gaussian_2d_fun(X[i], gamma, Y[i], scope)
        if RR is None:
            RR = R
        else:
            RR += R
        if (Y[i]==1):
            c = 'red'
        else:
            c = 'blue'
        ax2.contour(P,Q,R,5,colors=c,linewidths=[0.5,0.2],linestyles='dashed')

    ax1.plot_surface(P, Q, RR, cmap=cm.coolwarm)
    draw_2d(ax2, X, Y, True)


    # 绘制地标
    ax2.scatter(L[:,0], L[:,1], marker='.', color='y')

    plt.show()
    

if __name__=="__main__":
    # 生成原始样本
    X_raw = np.array([[0,0],[1,1],[0,1],[1,0]])
    Y = np.array([-1,-1,1,1])
    print("X 的原始值：")
    print(X_raw)
    print("Y 的原始值：")
    print(Y)
    
    # 建立地标
    scope1 = [-0.5,1.5,5, -0.5,1.5,5]
    L = create_landmark(scope1)
    #print("地标：\n",L)

    # 显示2D高斯函数
    gamma = 1
    scope2 = [-1.5,2.5,50, -1.5,2.5,50]
    show_gaussian_2d(gamma, X_raw, Y, scope2, L)

    # 生成样本的特征矩阵
    X_feature = Feature_matrix(X_raw, L, gamma)
    print("Feature Matrix=")
    print(np.round(X_feature,3))
    print("2 号样本的特征值矩阵：")
    print(np.round(X_feature[2].reshape(scope1[2],scope1[5]),3))

    # 线性分类
    C = 1
    model = linear_svc(X_feature, Y, C)

    # 可视化结果
    scope3 = [-0.5,1.5,50, -0.5,1.5,50]
    X1,X2,y_pred,prob = prediction(model, gamma, L, scope3)
    print(y_pred)
    print(prob)
    show_result(X1, X2, y_pred, X_raw, Y, prob, scope3)
