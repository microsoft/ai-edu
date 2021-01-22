def test_list_param(arr):
    for i in range(0, len(arr)):
        arr[i] = arr[i] * 2
    return


x_arr = [1, 2, 3, 4, 5]
print("Before function:", x_arr)

test_list_param(x_arr)
print("After function:", x_arr)