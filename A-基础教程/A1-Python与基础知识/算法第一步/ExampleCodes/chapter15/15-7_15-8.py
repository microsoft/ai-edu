from Utilities import swap


def partition_v2(arr, low, high):
    if low >= high:
        return -1

    pi = low
    li = low + 1
    ri = high

    while ri >= li:
        if arr[li] > arr[pi]:
            swap(arr, ri, li)
            ri -= 1
        else:
            li += 1

    pi = li - 1
    swap(arr, low, pi)
    return pi

# 下面是 代码15-8

arr = [3, 9, 7, 8, 2, 4, 1, 6, 5, 17]
p = partition_v2(arr, 0, len(arr) - 1)
print("pivot index is:", p)
print(arr)
