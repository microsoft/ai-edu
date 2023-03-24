import numpy as np
import RentCar_0_Data as carData

def calculate_matrix(n_states, data_array):
    # 定义一个 n x n 的数组（矩阵）
    P_counter = np.zeros((n_states, n_states))
    rows = data_array.shape[1]  # 获得记录行数
    for i in range(rows):   # 一共100行记录
        data_list = data_array[i].ravel().tolist()
        # 把 ABCD 变成 0123
        X = [ord(x)-65 for x in data_list]
        for i in range(len(X)-1):
            rent_from = X[i]
            return_to = X[i+1]
            # 对应位置计数加 1
            P_counter[rent_from, return_to] += 1
    #endfor
    # 计算各列之和
    sum = np.sum(P_counter, axis=1, keepdims=True)
    print("各个状态出现的次数:\n",sum)
    P = P_counter / sum
    return P

if __name__ == "__main__":
    # 状态空间
    n_states = 4
    # 读取文件
    data_array = carData.read_data()
    P = calculate_matrix(n_states, data_array)
    print("概率转移矩阵:")
    print(np.around(P, 1))
