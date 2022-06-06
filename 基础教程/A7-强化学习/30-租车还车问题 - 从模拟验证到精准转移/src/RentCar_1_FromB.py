import RentCar_0_Data as carData
from enum import Enum

class RentalStore(Enum):
    A = 0
    B = 1
    C = 2
    D = 3

# 统计rent_from出现的次数，以及经过t天转移到return_to的次数
def counter_from_t_to(X, rent_from, return_to, t):
    n_from = 0
    n_to = 0
    for x in range(len(X)-t):
        if rent_from == X[x]:
            n_from += 1
            if return_to == X[x+t]: # {t} 天后还到 {return_to} 店
                n_to += 1
    return n_from, n_to

def Statistic(data_array, rent_from, return_to, t=1):
    rows = data_array.shape[1]
    num_from = 0
    num_to = 0
    for i in range(rows):   # 一共100行记录
        data_list = data_array[i].ravel().tolist()
        # 把 ABCD 变成 0123
        X = [ord(x)-65 for x in data_list]
        # 统计 from->to 的总数
        n_from, n_to = counter_from_t_to(X, rent_from.value, return_to.value, t)
        num_from += n_from
        num_to += n_to
    #endfor
    return num_from, num_to

if __name__ == "__main__":
    # 读取文件
    data_array = carData.read_data()
    rent_from = RentalStore.B

    print("天数 =",2)
    for return_to in RentalStore:
        num_from, num_to = Statistic(data_array, rent_from, return_to, t=2)
        print(str.format("从 {0} 店出租 {2} 次，还到 {1} 店 {3} 次", rent_from.name, return_to.name, num_from, num_to))

    print("天数 =",5)
    for return_to in RentalStore:
        num_from, num_to = Statistic(data_array, rent_from, return_to, t=5)
        print(str.format("从 {0} 店出租 {2} 次，还到 {1} 店 {3} 次", rent_from.name, return_to.name, num_from, num_to))

    print("天数 =",1)
    for return_to in RentalStore:
        num_from, num_to = Statistic(data_array, rent_from, return_to, t=1)
        print(str.format("从 {0} 店出租 {2} 次，还到 {1} 店 {3} 次", rent_from.name, return_to.name, num_from, num_to))
