arr = list(range(1, 1001))  # 生成一个List，里面按顺序存储了1~1000这1000个元素
tn = 635  # 可以随便改

low = 0
high = len(arr) - 1

while low <= high:
    m = int((high - low) / 2) + low

    if arr[m] == tn:
        # 把打印出目标数所在的位置下标改成直接打印出目标数
        print("Succeeded! The target number is: ", arr[m])
        break
    else:
        if arr[m] < tn:
            low = m + 1
        else:
            high = m - 1

    if low > high:
        print("Search failed.")