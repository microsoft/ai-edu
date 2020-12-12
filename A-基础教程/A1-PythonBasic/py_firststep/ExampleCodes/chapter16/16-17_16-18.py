from Utilities import partition
from Utilities import partition_v2
from Utilities import generate_test_data


def qsort_recursion_v1(arr, low, high):

    if low >= high:
        return
    print("V1", low, high, arr)

    p = partition(arr, low, high)  # 调用新的分区函数
    qsort_recursion_v1(arr, low, p - 1)
    qsort_recursion_v1(arr, p + 1, high)

    return


def qsort_recursion_v2(arr, low, high):
    if low >= high:
        return
    print("V2", low, high, arr)

    p = partition_v2(arr, low, high)  # 调用新的分区函数
    qsort_recursion_v2(arr, low, p - 1)
    qsort_recursion_v2(arr, p + 1, high)

    return

# 下面为16-18

start = 1
end = 10
_, _, arr_reverse = generate_test_data(start, end)
qsort_recursion_v1(arr_reverse, 0, len(arr_reverse) - 1)

print("\n-----\n")

_, _, arr_reverse = generate_test_data(start, end)
qsort_recursion_v2(arr_reverse, 0, len(arr_reverse) - 1)