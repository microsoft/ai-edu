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
    for i in range(n_samples):
        next = np.random.choice(states, p=P[current])
        X.append(next)
        current = next
    #endfor
    return X

def save_file(X, file_name):
    # 把0123变成ABCD
    Y = [chr(x+65) for x in X]
    #print(Y)
    # 保存Y到文件
    json_list = json.dumps(Y)
    file = open(file_name, "w")
    file.write(json_list)
    file.close()


if __name__ == "__main__":
    # 采样数量
    n_samples = 10000
    # 状态空间
    n_states = 4
    # 起始状态(从0开始)
    start_state = 1
    X = sample(n_samples, n_states, start_state)
    #print(X)
    # 保存文件
    root = os.path.split(os.path.realpath(__file__))[0]
    file_name = os.path.join(root, "CarData.txt")
    save_file(X, file_name)
