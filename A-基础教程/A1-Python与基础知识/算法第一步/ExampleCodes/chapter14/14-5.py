from Utilities import swap


def bubble_sort(arr):
    for i in range(0, len(arr) - 1):
        swapped = False

        for j in range(len(arr) -1, i, -1):
            if arr[j] <arr[j - 1]:
                swap(arr, j, j-1)
                swapped = True

        if not swapped:
            return
    return

# 下面是调用代码 无需出现在书中
arr = [3, 9, 4, 11, 7, 2, 4, 1, 0, 15, 23, 43, 38, 17]
bubble_sort(arr)
print(arr)