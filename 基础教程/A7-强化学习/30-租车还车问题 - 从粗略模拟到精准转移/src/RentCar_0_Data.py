import numpy as np
import os
import json

# 作为目标的转移概率矩阵
P = np.array([
    [0.1, 0.3, 0.0, 0.6],
    [0.8, 0.0, 0.2, 0.0],
    [0.0, 0.9, 0.1, 0.0],
    [0.0, 0.3, 0.3, 0.4]
])

# 采样
def sample(n_samples, n_states, start_state):
    states = [i for i in range(n_states)]
    # 状态转移序列
    X = []
    # 开始采样
    X.append(start_state)
    current = start_state
    for i in range(n_samples-1): # start_state也算一个
        next = np.random.choice(states, p=P[current])
        X.append(next)
        current = next
    #endfor
    return X

def get_fullpath():
    root = os.path.split(os.path.realpath(__file__))[0]
    file_name = os.path.join(root, "CarData.txt")
    return file_name

def save_data(data_array):
    file_name = get_fullpath()
    np.savetxt(file_name, data_array, fmt='%s')

def read_data():
    file_name = get_fullpath()
    data_array = np.loadtxt(file_name, dtype='<U1') # 以字符形式存储
    #print(data_array.shape)
    return data_array

if __name__ == "__main__":
    # 采样数量
    n_samples = 10000
    # 状态空间
    n_states = 4
    # 起始状态(0 based)
    start_state = 1
    X = sample(n_samples, n_states, start_state)
    # 把0123变成ABCD
    Y = [chr(x+65) for x in X]
    data_array = np.reshape(Y, (100,100))
    # 保存文件
    save_data(data_array)
