def partition(arr):
    if len(arr) < 2:  # 待分区数列长度为1或0
        # 直接返回其本身作为左分区，再返回一个空的轴和一个空List作为右分区
        return arr, None, None

    left_partition = []
    right_partition = []
    pivot = arr[0]  # 将当前数列中的第一个元素作为“轴”

    for i in range(1, len(arr)):
        if arr[i] <= pivot:
            left_partition.append(arr[i])  # 小于或等于轴的元素放到左分区
        else:
            right_partition.append(arr[i])  # 大于轴的元素放到右分区

    return left_partition, pivot, right_partition  # 按顺序返回左分区、轴和右分区


# 下面是 代码15-2

arr = [3, 9, 7, 8, 2, 4, 1, 6, 5, 17]
print(partition(arr))
