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

# 下面一行为打印提示语句，不要出现在代码示例里：
print("\n\nFollowing are output of code 10-3\n")

# 下面代码要显示在书中10-3部分
arr = [3, 5, 9, 7, 12, 15, 18, 32, 66, 78, 94, 103, 269]
tn = 5

result = binary_search(arr, tn)

if result >= 0:
    print("Succeeded! The target index is: ", result)
else:
    print("Search failed.")


# 下面一行为打印提示语句，不要出现在代码示例里：
print("\n\nFollowing are output of code 10-4\n")  # TOBE IGNORED

# 下面代码要显示在书中10-4部分
arr = []
for i in range(1, 1001):
    arr.append(i)

for tn in range(1, 1001):
    result = binary_search(arr, tn)

    if result >= 0:
        print("Succeeded! The target index is: ", result)
    else:
        print("Search failed.")