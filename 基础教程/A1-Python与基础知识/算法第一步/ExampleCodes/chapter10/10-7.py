from SearchAlgorithms import binary_search

while 1:
    arr_input = input("input the number sequence, separated by ',':")
    arr_strs = arr_input.strip().split(',')

    arr = list(map(int, arr_strs))

    tn_input = input("input target number:")
    tn = int(tn_input.strip())

    result = binary_search(arr, tn)

    if result >= 0:
        print("Succeeded! The target index is: ", result)
    else:
        print("Search failed.")