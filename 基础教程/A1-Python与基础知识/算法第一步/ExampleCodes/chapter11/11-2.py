tn = 635  # 可以随便改

low = 1
high = 1000

while low <= high:
    m = int((high - low) / 2) + low
    if m == tn:
        # 打印出目标数
        print("Succeeded! The target number is: ", m)
        break
    else:
        if m < tn:
            low = m + 1
        else:
            high = m - 1

if low > high:
    print("Search failed.")