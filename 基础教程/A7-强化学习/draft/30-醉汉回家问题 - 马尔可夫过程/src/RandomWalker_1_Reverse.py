import numpy as np

# 从酒馆出发,允许向家的相反方向走
def RandomWalker(distance=50):
    position = 0    # 距离酒馆的位置，为50时表示到家
    counter = 0     # 行走的步数（包括原地不动）
    trajectory = [] # 行走的路径
    while(position < distance):
        # 随机选择向前(1)向后(-1)不动(0), 概率是[0.4,0.2,0.4]
        step = np.random.choice([-1,0,1], p=[0.4,0.2,0.4])
        position += step    # 更新位置
        counter += 1        # 更新步数
        trajectory.append(position) # 记录位置

    # 输出最后位置和步数，计算位置平均值
    print(str.format("步数 : {0}\t最远 : {1}\t平均步数 : {2}", counter, np.min(trajectory), np.mean(trajectory)))

if __name__ == "__main__":
    for i in range(10):
        RandomWalker(50)
