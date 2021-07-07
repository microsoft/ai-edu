def partition(arr):
    if len(arr) < 2:
        return -1

    left_partition = []
    right_partition = []
    pivot = arr[0]  # 将当前数列中的第一个元素作为“轴”

    for i in range(1, len(arr)):
        if arr[i] <= pivot:
            left_partition.append(arr[i])  # 小于等于轴的元素放到左分区
        else:
            right_partition.append(arr[i])  # 大于轴的元素放到右分区

    llen = len(left_partition)
    arr[0:llen] = left_partition[0:llen]
    arr[llen] = pivot
    arr[llen + 1: len(arr)] = right_partition[0:len(right_partition)]
    return llen


# 下面是 代码15-4

arr = [3, 9, 7, 8, 2, 4, 1, 6, 5, 17]
p = partition(arr)
print("pivot index is:", p)
print(arr)
