from Utilities import partition_v2


def qsort_recursion(arr, low, high):
    if low >= high:
        return
    p = partition_v2(arr, low, high)
    qsort_recursion(arr, low, p - 1)
    qsort_recursion(arr, p + 1, high)
    return


# 下面是 代码16-11
arr = [7, 9, 6, 8, 10, 3, 2, 1, 4, 5]
qsort_recursion(arr, 0, len(arr) - 1)
print(arr)
