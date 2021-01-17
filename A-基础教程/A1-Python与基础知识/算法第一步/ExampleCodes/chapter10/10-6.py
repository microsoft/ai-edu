from SearchAlgorithms import binary_search

arr_input = input("input the number sequence, separated by ',':")

arr_strs = arr_input.strip().split(',')  # 输入的序列以逗号为分割，切分成一个List的若干元素

arr = list(map(int, arr_strs))  # 将一个元素类型为字符串的序列转换为类型为整型的序列

tn_input = input("input target number:")

tn = int(tn_input.strip())

result = binary_search(arr, tn)

if result >= 0:
    print("Succeeded! The target index is: ", result)
else:
    print("Search failed.")