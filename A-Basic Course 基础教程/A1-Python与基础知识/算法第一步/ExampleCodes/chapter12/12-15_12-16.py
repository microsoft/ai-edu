def binary_search_in_rotated_sequence(arr, tn):
    low = 0
    high = len(arr)-1

    while low <= high :
        m = int((high - low)/2) + low
        if arr[m] == tn:
            return m
        else:
            if arr[m] < tn:
                if arr[m] <arr[low]:
                    if arr[low] <= tn:
                        high = m - 1
                    else:
                        low = m + 1
                else:
                    low = m + 1
            else:  # arr[m] > tn
                if arr[m] >arr[high]:
                    if arr[high] >= tn:
                        low = m + 1
                    else:
                        high = m - 1
                else:
                    high = m - 1

    if low > high:
        return -1

# 下面为代码12-16

arr = []
for i in range(11, 21):
    arr.append(i)

for i in range(1, 11):
    arr.append(i)

tnList = [1, 2, 7, 10, 11, 15, 20]

for tn in tnList:
    r = binary_search_in_rotated_sequence(arr, tn)
    if r == -1:
        print("Failed to search", tn)
    else:
        print(tn, "is found in position", r)