# 本文件与10-2完全一致


def binary_search(arr, tn):
    low = 0
    high = len(arr) - 1

    while low <= high:
        m = int((high - low) / 2) + low
        if arr[m] == tn:
            return m
        else:
            if arr[m] < tn:
                low = m + 1
            else:
                high = m - 1

    if low > high:
        return -1