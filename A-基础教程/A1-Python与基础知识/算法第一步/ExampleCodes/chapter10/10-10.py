def test_list_param(arr):
    for i in range(0, len(arr)):
        arr[i] = arr[i] * 2
    return arr


x_arr = [1, 2, 3, 4, 5]
y_arr = test_list_param(x_arr)

print("xArr is", x_arr)
print("yArr is", y_arr)