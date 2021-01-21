# binary_search与10-2完全一致，这个函数不用出现在代码示例里

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

# 下面为12-1正式代码

arr = [3, 5, 5, 5, 5, 9, 7, 12, 15, 18, 32, 66, 78, 94, 103, 269]
tn = 5

result = binary_search(arr, tn)

if result >= 0:
    print("Succeeded! The target index is: ", result)
else:
    print("Search failed.")