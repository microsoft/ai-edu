arr = ["apple", "orange", "watermelon"]

i = 0
while i < len(arr):
    print(arr[i])
    if arr[i] == "orange":
        break;
    i = i + 1

if i == len(arr):
    print("No more fruit.")