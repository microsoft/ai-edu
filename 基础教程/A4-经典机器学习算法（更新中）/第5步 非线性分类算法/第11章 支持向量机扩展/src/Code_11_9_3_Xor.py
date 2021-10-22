
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import *
from sklearn.preprocessing import *
from sklearn.svm import *
import matplotlib as mpl
from matplotlib.colors import ListedColormap

def linear_svc(X,Y,C):
    model = SVC(C=C, kernel='linear')
    model.fit(X,Y)

    print("权重:",model.coef_)
    print("支持向量个数:",model.n_support_)
    print("支持向量索引:",model.support_)
    print("支持向量:",np.round(model.support_vectors_,3))
    print("支持向量ay:",model.dual_coef_)
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
    X12_new = K_matrix(X12, landmark, gamma)
    # 做预测
    pred = model.predict(X12_new)
    # 变形并绘制分类区域
    y_pred = pred.reshape(X1.shape)

    return X1, X2, y_pred

# 展示分类结果
def show_result(X1, X2, y_pred, X_sample, Y):
    # 基本绘图设置
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.title(u"异或问题的分类结果")
    plt.grid()
    plt.axis('equal')
    # 绘图
    cmap = ListedColormap(['yellow','lightgray'])
    plt.contourf(X1,X2, y_pred, cmap=cmap)
    # 绘制原始样本点用于比对
    draw_2d(plt, X_sample, Y)

    plt.show()

def draw_2d(ax, x, y):
    ax.scatter(x[y==1,0], x[y==1,1], marker='^')
    ax.scatter(x[y==-1,0], x[y==-1,1], marker='o')

# 映射成核矩阵
# X - 样本数据
# L - 地标 Landmark，在此例中就是样本数据
def K_matrix(X, L, gamma):
    n = X.shape[0]  # 样本数量
    m = L.shape[0]  # 特征数量
    K = np.zeros(shape=(n,m))
    for i in range(n):
        for j in range(m):
            # 计算每个样本点到其它样本点之间的高斯核函数值
            K[i,j] = np.exp(-gamma * np.linalg.norm(X[i] - L[j])**2)

    return K

if __name__=="__main__":
    # 生成原始样本
    X_raw = np.array([[0,0],[1,1],[0,1],[1,0]])
    Y = np.array([-1,-1,1,1])
    print("X 的原始值：")
    print(X_raw)
    print("Y 的原始值：")
    print(Y)
    
    # 标准化
    ss = StandardScaler()
    X = ss.fit_transform(X_raw)
    print("X 标准化后的值：")
    print(X)

    # 用 K 函数做映射，形成核函数矩阵
    gamma = 1
    X_new = K_matrix(X, X, gamma)
    print("映射结果：")
    print(np.round(X_new,3))

    # 尝试用线性 SVM 做分类    
    C = 1
    model = linear_svc(X_new, Y, C)
    # 显示分类预测结果
    X1, X2, y_pred = prediction(model, gamma, X, [-1.5,1.5,100,-1.5,1.5,100])
    show_result(X1, X2, y_pred, X, Y)
