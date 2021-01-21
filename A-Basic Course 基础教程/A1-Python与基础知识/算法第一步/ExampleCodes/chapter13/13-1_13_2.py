def swap(arr, i, j):
    if len(arr) < 2:
        return

    if i < 0 or i >= len(arr) or j < 0 or j >= len(arr):
        return

    if i == j:
        return

    tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp

    return


# 下面为代码13-2

arr = [3, 2, 1, 5, 8, 7, 9, 10, 13]
print(arr)
swap(arr, 0, 1)
print(arr)
