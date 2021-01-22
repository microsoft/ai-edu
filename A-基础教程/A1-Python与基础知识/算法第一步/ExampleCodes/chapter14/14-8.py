from Utilities import swap


def selection_sort(arr):
    for i in range(0, len(arr)):
        min_position = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_position]:
                min_position = j
                swap(arr, i, min_position)
    return


def bubble_sort(arr):
    for i in range(0, len(arr) - 1):
        swapped = False

        for j in range(len(arr) -1, i, -1):
            if arr[j] <arr[j - 1]:
                swap(arr, j, j-1)
                swapped = True

        if not swapped:
            return
    return


def insertion_sort(arr):
    if len(arr) == 1:
        return
    for i in range(1, len(arr)):
        for j in range(i, 0, -1):
            if arr[j] < arr[j - 1]:
                swap(arr, j, j - 1)
            else:
                break
    return