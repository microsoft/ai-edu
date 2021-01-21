tn = 95
arr = [1, 5, 8, 19, 3, 2, 14, 6, 8, 22, 44, 95, 78]

i = 0
while i < len(arr):
    if arr[i] == tn:
        print("tn is element", i)
        break
    else:
        i = i + 1

if i == len(arr):
    print("failed")