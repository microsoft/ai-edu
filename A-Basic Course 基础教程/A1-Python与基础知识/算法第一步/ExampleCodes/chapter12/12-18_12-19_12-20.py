def binary_search_in_rotated_repeating_sequence(arr, tn, delta):
    low = 0
    high = len(arr) - 1

    if delta < 0 and arr[0] == tn:
        return 0

    if delta > 0 and arr[len(arr) -1] == tn:
        return len(arr) -1

    while low <= high :
        m = int((high - low)/2) + low

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

# 下面一行为打印提示语句，不要出现在代码示例里：
print("\n\nFollowing are output of code 12-19\n")  # TOBE IGNORED

# 下面为代码12-19

arr = [10]

for i in range(11, 21):
    arr.append(i)
    arr.append(i)

for i in range(1, 11):
    arr.append(i)
    arr.append(i)

tnList = [1, 2, 7, 10, 11, 15, 20]

for tn in tnList:
    r = binary_search_in_rotated_repeating_sequence(arr, tn, -1)
    if r == -1:
        print("Failed to search", tn)
    else:
        print(tn, "is found in position", r)

# 下面一行为打印提示语句，不要出现在代码示例里：
print("\n\nFollowing are output of code 12-20\n")

# 下面为代码12-20

arrList = [[2,1,1,1,1], [1], [2,1,2,2,2,2,2], [5,6,1,2,3,4], [1,2,1,1,1,1],[1,2,2,3,3,3,4,5,6,6,7,1]]
tn = 1

for arr in arrList:
    r = binary_search_in_rotated_repeating_sequence(arr, tn, -1)
    if r == -1:
        print("Failed to search", tn)
    else:
        print(tn, "is found in position", r)