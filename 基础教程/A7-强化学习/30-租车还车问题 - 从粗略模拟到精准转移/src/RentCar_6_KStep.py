import numpy as np

P = np.array([
    [0.1, 0.3, 0.0, 0.6],
    [0.8, 0.0, 0.2, 0.0],
    [0.0, 0.9, 0.1, 0.0],
    [0.0, 0.3, 0.3, 0.4]
])

# 计算K步转移概率矩阵        
def K_step_matrix(P, K):
    Pk=P.copy()
    for i in range(K-1):
        Pk=np.dot(P,Pk)
        #print(Pk)
    return Pk

if __name__=="__main__":
    X = np.array([0,1,0,0])
    P5 = K_step_matrix(P, 5)
    print("5步转移矩阵:\n", P5)
    X5 = np.dot(X, P5)
    print("第 5 天的情况：", X5)
