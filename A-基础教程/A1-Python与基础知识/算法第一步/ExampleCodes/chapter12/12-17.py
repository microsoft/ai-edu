def binary_search_in_rotated_sequence(arr, tn):
    low = 0
    high = len(arr) - 1

    while low <= high :
        m = int((high - low)/2) + low

        if arr[m] == tn:
            return m
        else:
            if arr[m] < tn:
                if arr[m] < arr[low] <= tn:
                    high = m - 1
                else:
                    low = m + 1
            else:
                if arr[m] > arr[high] >= tn:
                    low = m + 1
                else:
                    high = m - 1

    if low > high:
        return -1