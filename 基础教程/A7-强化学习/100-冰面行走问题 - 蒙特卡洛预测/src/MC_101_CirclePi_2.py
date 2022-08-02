import tqdm
import matplotlib.pyplot as plt
import numpy as np


# 随机铺点计算 pi 值
def put_points(num_total_points):
    data = np.random.uniform(-1, 1, size=(num_total_points, 2))     # 在正方形内生成随机点
    r = np.sqrt(np.power(data[:,0], 2) + np.power(data[:,1], 2))    # 计算每个点到中心的距离
    r[r<=1]=1
    r[r>1]=0
    num_in_circle = np.sum(r)
    pi = num_in_circle/num_total_points*4
    return pi

if __name__=="__main__":
    np.random.seed(15)
    pis = []
    # 取 1000,1100,1200,...,20000个点做试验
    for n in tqdm.trange(1000,20000,100):
        pi = 0
        # 重复100次求平均
        for j in range(100):
            pi += put_points(n)
        pis.append(pi/100)  # 求平均
    
    plt.grid()
    plt.plot(pis)
    plt.plot([0,200],[3.14159265,3.14159265])   # 画基准线
    plt.title(str.format("average={0}", np.mean(pis)))
    plt.show()
