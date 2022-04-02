import RentCar_0_Data as carData
from enum import Enum

class RentalStore(Enum):
    A = 0
    B = 1
    C = 2
    D = 3

# 统计ai出现的次数，以及从ai经过t天转移到aj的次数
def counter_t(X, rent_from, return_to, t):
    ni = 0
    nj = 0
    for x in range(len(X)-t):
        if rent_from == X[x]:
            ni += 1
            if return_to == X[x+t]:
                nj += 1
    return ni, nj

def P_from_t_to(data_array, rent_from, return_to, t=1):
    rows = data_array.shape[1]
    num_from = 0
    num_to = 0
    for i in range(rows):   # 一共100行记录
        data_list = data_array[i].ravel().tolist()
        # 把 ABCD 变成 0123
        X = [ord(x)-65 for x in data_list]
        ni, nj = counter_t(X, rent_from.value, return_to.value, t)  # 0代表A, 1代表B, 即从A租，到B还
        num_from += ni
        num_to += nj
    #endfor
    print(str.format("{0}->{1} : {2},{3}", rent_from.name, return_to.name, num_from, num_to))
    # 打开这一行的注释来观察具体概率数据
    #print(num_to/num_from)

if __name__ == "__main__":
    # 读取文件
    data_array = carData.read_data()
    rent_from = RentalStore.B

    print("天数 =",2)
    for return_to in RentalStore:
        P_from_t_to(data_array, rent_from, return_to, t=2)

    print("天数 =",5)
    for return_to in RentalStore:
        P_from_t_to(data_array, rent_from, return_to, t=5)
