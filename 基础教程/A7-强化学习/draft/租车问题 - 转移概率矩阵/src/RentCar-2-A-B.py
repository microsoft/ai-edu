import RentCar_0_Data as carData

# 统计ai出现的次数，以及从ai转移到aj的次数
def counter_1(X, rent_from, return_to):
    ni = 0
    nj = 0
    for x in range(len(X)-1):
        if rent_from == X[x]:
            ni += 1
            if return_to == X[x+1]:
                nj += 1
    return ni, nj

def P_A_B(data_array):
    rows = data_array.shape[1]
    num_from = 0
    num_to = 0
    for i in range(rows):   # 一共100行记录
        data_list = data_array[i].ravel().tolist()
        # 把 ABCD 变成 0123, 便于计算
        X = [ord(x)-65 for x in data_list]
        ni, nj = counter_1(X, 0, 1)  # 0代表A, 1代表B, 即从A租，到B还
        num_from += ni
        num_to += nj
    #endfor
    print(num_from, num_to, num_to/num_from)



if __name__ == "__main__":
    # 读取文件
    data_array = carData.read_data()
    P_A_B(data_array)
