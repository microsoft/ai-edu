arr = [3, 5, 9, 7, 12, 15, 18, 32, 66, 78, 94, 103, 269]  # 例表内数值可以随便改，只要保证有序排列即可
tn = 5  # 可以随便改，arr中有没有都可以

low = 0
high = len(arr) - 1

while low <= high:
    m = int((high - low)/2) + low
    if arr[m] == tn:
        print("Succeeded! The target index is: ", m)
        break
    else:
        if arr[m] < tn:
            low = m + 1
        else:
            high = m - 1

if low > high:
    print("Search failed.")