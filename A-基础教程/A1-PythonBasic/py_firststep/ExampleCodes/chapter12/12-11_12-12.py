def quick_repeating_sequence_binary_search(arr, tn, delta):
    low = 0
    high = len(arr) - 1

    while low <= high:
        m = int((high - low) / 2) + low

        if arr[m] == tn:
            if 0 <= m + delta < len(arr) and arr[m + delta] == tn:
                if delta < 0:
                    high = m - 1
                else:
                    low = m + 1
            else:
                return m
        else:
            if arr[m] < tn:
                low = m + 1
            else:
                high = m - 1

        if low > high:
            return -1


# 下面为代码12-12

arr = [3, 3, 3, 5, 5, 5, 5, 9, 7, 12, 15, 18, 32, 66, 78, 94, 103, 269]
tn = 5

result = quick_repeating_sequence_binary_search(arr, tn, -1)

if result >= 0:
    print("Succeeded! The target index is: ", result)
else:
    print("Search failed.")
