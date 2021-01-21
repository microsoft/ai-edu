def partition(arr, low, high):
    if low >= high:
        return -1

    left_partition = []
    right_partition = []
    pivot = arr[low]

    for i in range(low + 1, high + 1):
        if arr[i] <= pivot:
            left_partition.append(arr[i])
        else:
            right_partition.append(arr[i])

    llen = len(left_partition)
    rlen = len(right_partition)

    for i in range(0, llen):
        arr[i + low] = left_partition[i]
        arr[low + llen] = pivot

    for i in range(0, rlen):
        arr[i + low + llen + 1] = right_partition[i]

    return low + llen


# 下面是 代码15-6

arr = [3, 9, 7, 8, 2, 4, 1, 6, 5, 17]
p = partition(arr, 0, len(arr) - 1)
print("pivot index is:", p)
print(arr)
