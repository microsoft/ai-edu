
import numpy as np
from sklearn.datasets import *
from sklearn.preprocessing import *

# 理论上的映射函数，但实际上不能用
def mapping_function(X, sigma):
    n = X.shape[0]
    Z = np.zeros((n, 4))    # 做一个4维的特征映射，即式10中的n=0,1,2,3
    for i in range(n):
        # 求 x 矢量的模，是一个标量, 式 11.8.6
        z_norm = np.linalg.norm(X[i])
        # 第 0 维 = exp(-0.5*z^2/sigma^2)
        Z[i,0] = np.exp(-0.5*(z_norm**2)/sigma**2)
        # 第 1 维
        Z[i,1] =  Z[i,0] * np.sqrt(2) * (z_norm  / sigma)
        # 第 2 维
        Z[i,2] = Z[i,0] * np.sqrt(2**2/2) * (z_norm / sigma)**2
        # 第 3 维
        Z[i,3] = Z[i,0] * np.sqrt(2**3/6) * (z_norm / sigma)**3

    return Z


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

    # X 标准化后映射的特征值
    sigma = 2
    Z1 = mapping_function(X, sigma)
    print("X 标准化后映射的特征值：")
    print(Z1)
    # 通过结果可以看出来映射后4个样本被映射到了四维空间中的一个点，不能做后续的分类
   
    # X 不做标准化直接做映射的特征值
    sigma = 2
    Z2 = mapping_function(X_raw, sigma)
    print("X 不做标准化直接做映射的特征值：")
    print(Z2)

