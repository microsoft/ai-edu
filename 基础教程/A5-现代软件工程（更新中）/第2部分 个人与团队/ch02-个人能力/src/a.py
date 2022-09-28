
import random

def generate_data():
    data = [i for i in range(1,101)]
    data.append(random.randint(1,100))
    print(data)
    random.shuffle(data)
    print(data)
    return data

def method_1_sum(data):
    sum = 0
    for x in data:
        sum += x
    print(sum - 5050)

def method_2_dict(data):
    dict = {}
    for x in data:
        if dict.__contains__(x):
            print(x)
            return
        else:
            dict[x]=1

def method_3_sort(data):
    tmp = sorted(data)
    for i in range(101):
        if tmp[i] != i+1:
            print(i)
            return

def method_4_search(data):
    pos = 0
    while (True):
        if (pos == data[pos]):
            print(pos)
            break

        x = data[pos]
        pos = data[x]
        data[x] = x

    print(data)

if __name__ == "__main__":
    data = generate_data()
    method_1_sum(data)
    method_2_dict(data)
    method_3_sort(data)
    method_4_search(data)
