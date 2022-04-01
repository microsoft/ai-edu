import numpy as np
import RentCar_0_Data as carData

def calculate_Matrix(n_states, data_array):
    P_counter = np.zeros((n_states, n_states))
    rows = data_array.shape[1]
    for i in range(rows):   # 一共100行记录
        data_list = data_array[i].ravel().tolist()
        # 把 ABCD 变成 0123
        X = [ord(x)-65 for x in data_list]
        for i in range(len(X)-1):
            ai = X[i]
            aj = X[i+1]
            P_counter[ai, aj] += 1
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
    P = calculate_Matrix(n_states, data_array)
    print("概率转移矩阵:")
    print(np.around(P, 1))
