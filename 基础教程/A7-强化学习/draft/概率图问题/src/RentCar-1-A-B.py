import RentCar_0_Data as carData

# 统计ai出现的次数，从ai转移到aj的次数
def counter(X, ai, aj):
    ni = 0
    nj = 0
    for x in range(len(X)-1):
        if ai == X[x]:
            ni += 1
            if aj == X[x+1]:
                nj += 1
    return ni, nj

if __name__ == "__main__":
    # 读取文件
    data_array = carData.read_data()
    rows = data_array.shape[1]
    num_i = 0
    num_j = 0
    for i in range(rows):   # 一共100行记录
        data_list = data_array[i].ravel().tolist()
        # 把 ABCD 变成 0123
        X = [ord(x)-65 for x in data_list]
        ni, nj = counter(X, 0, 1)  # 0代表A, 1代表B, 即从A租，到B还
        num_i += ni
        num_j += nj
    #endfor
    print(num_i, num_j, num_j/num_i)
