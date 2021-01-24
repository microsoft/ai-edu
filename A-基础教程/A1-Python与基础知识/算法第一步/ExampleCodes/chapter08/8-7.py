tn = 95
arr = [1, 5, 8, 19, 3, 2, 14, 6, 8, 22, 44, 95, 78]

for i in range(0, len(arr)):
    if arr[i] == tn:
        print("tn is element", i)
        break

if i == len(arr):
    print("failed")