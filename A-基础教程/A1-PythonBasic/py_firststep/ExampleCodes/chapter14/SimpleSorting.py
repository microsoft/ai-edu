from Utilities import swap


def selection_sort(arr):
    #startPosition是本次迭代的起始位置下标，与前述步骤中的k相对应：startPosition == k - 1
    for start_position in range(0, len(arr)):
        min_position = start_position # minPosition用来记录本次迭代中最小数值所在位置下标

        #和其后所有位置上的数字比较，如果有更小的数字，则用该位置替代当前的minPosition
        for i in range(start_position+1, len(arr)):
            if (arr[i] <arr[min_position]):
                min_position = i

        #经过一轮比较，当前的minPosition已经是当前待排序数字中的最小值，将它和本次迭代第一个位置上的数字交换
        swap(arr, start_position, min_position)
    return


def bubble_sort(arr):
    for i in range(0, len(arr) - 1):
        for j in range(len(arr) -1, i, -1):
            if arr[j] <arr[j - 1]:
                swap(arr, j, j-1)
    return


def insertion_sort(arr):
    if (len(arr) == 1):#因为要从第一个元素之后的元素迭代，所以如果整个序列长度为1，则直接返回
        return

    for i in range(1, len(arr)):
        # 此处也是倒着访问List，但不是从尾巴开始的，而是从当前位置开始的，因为是两两交换，所以此处代码与bubbleSort有些相似
        for j in range(i, 0, -1):
            if arr[j] <arr[j - 1]:
                swap(arr, j, j - 1)
            else:
                break
    return