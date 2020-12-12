from Utilities import swap


def insertion_sort(arr):
    if len(arr) == 1:  # 因为要从第一个元素之后的元素迭代，所以如果整个序列长度为1，则直接返回
        return

    for i in range(1, len(arr)):
        # 此处也是倒着访问List，但不是从尾巴开始的，而是从当前位置开始的，因为是两两交换，所以此处代码与bubbleSort有些相似
        for j in range(i, 0, -1):
            if arr[j] < arr[j - 1]:
                swap(arr, j, j - 1)
            else:
                break
    return


# 下面是代码 14-7

arr = [2, 1, 5, 8, 7, 13]
insertion_sort(arr)
print(arr)