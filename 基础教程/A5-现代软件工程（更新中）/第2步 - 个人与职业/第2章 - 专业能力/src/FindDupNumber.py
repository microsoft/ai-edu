
import random

max_number = 10     # 以最大值为10为例，便于说明问题，读者可以自行改到100

def generate_data():
    data = [i for i in range(1, max_number+1)]  # 生成一个顺序数组
    data.append(random.randint(1, max_number))  # 最后随机填一个数
    print("顺序数组=",data)
    random.shuffle(data)                        # 打乱顺序
    print("乱序数组=", data)
    return data

def method_1_sum(data):
    sum = 0
    for x in data:
        sum += x
    #print(sum - 5050)
    print(sum - int((1+max_number)*max_number/2))

def method_2_dict(data):
    dict = {}
    for x in data:
        if dict.__contains__(x):    # 字典中已经有此数，是重复的
            print(x)
            return
        else:                       # 如果字典中没有此数，则保存
            dict[x]=1   

def method_3_sort(data):
    data.sort()  
    for i in range(max_number+1):
        if data[i] == data[i+1]:    # 相邻的两个数相等，是重复数字
            print(i+1)
            return

def method_4_search(data):
    pos = 0                 # 从 0 位开始
    x = data[pos]           # 从 0 位取出数字
    while (True):
        if (x == pos):      # 在目标位置上已经有一个相同的数字，是重复的
            data[0] = x
            print(x)
            break
        pos = x             # 保存 x 的值到 pos，即
        x = data[pos]       # 取出 pos 位置的数值 x
        data[pos] = pos     # 把 pos 位置成 pos 值，如，第 3 个数组单元就置成 3

    print(data)


def method_5_xor(data):
    # 求所有数字的异或结果
    tmp_x = 0
    for i in range(len(data)):
        tmp_x = tmp_x ^ data[i]
    print(tmp_x)
    # 求 1~max_number 的异或结果
    tmp_n = 0
    for i in range(max_number):
        tmp_n = tmp_n ^ (i+1)   # 注意是 i+1，不是 i
    print(tmp_n)
    # 上面两者异或，可以得到重复的数字
    print(tmp_x ^ tmp_n)

if __name__ == "__main__":
    data = generate_data()
    method_1_sum(data.copy())
    method_2_dict(data.copy())
    method_3_sort(data.copy())
    method_4_search(data.copy())
    method_5_xor(data.copy())
