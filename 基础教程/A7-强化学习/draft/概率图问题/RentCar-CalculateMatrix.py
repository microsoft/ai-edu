import numpy as np
import json
import os

def calculate_Matrix(n_states, X):
    P_counter = np.zeros((n_states, n_states))
    for i in range(len(X)-1):
        a_i = X[i]
        a_j = X[i+1]
        P_counter[a_i, a_j] += 1
    #endfor
    # 计算各列之和
    sum = np.sum(P_counter, axis=1, keepdims=True)
    print("各个状态出现的次数:\n",sum)
    P = P_counter / sum
    return P

def open_file(file_name):
    file = open(file_name, "r")
    lines = file.read()
    file.close()
    data_list = json.loads(lines)
    return data_list

if __name__ == "__main__":
    # 状态空间
    n_states = 4
    # 读取文件
    root = os.path.split(os.path.realpath(__file__))[0]
    file_name = os.path.join(root, "CarData.txt")
    data_list = open_file(file_name)
    # 把 ABCD 变成 0123
    X = [ord(x)-65 for x in data_list]
    # 计算转移矩阵
    P = calculate_Matrix(n_states, X)
    print("概率转移矩阵:")
    print(np.around(P, 1))
