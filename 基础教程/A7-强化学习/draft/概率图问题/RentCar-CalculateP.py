import numpy as np
import json
import os

# 计算从a_i转移到a_j的概率
def calculate_P(X, a_i, a_j):
    n_i = 0
    n_j = 0
    for x in range(len(X)-1):
        if a_i == X[x]:
            n_i += 1
            if a_j == X[x+1]:
                n_j += 1
    print(n_i, n_j, n_j/n_i)

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
    calculate_P(X, 0, 1)    # 1代表B店，0代表A店


