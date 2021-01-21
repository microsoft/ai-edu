from Utilities import swap


def selection_sort(arr):
    # startPosition是本次迭代的起始位置下标，与前述步骤中的k相对应：startPosition == k - 1
    for start_position in range(0, len(arr)):
        min_position = start_position # minPosition用来记录本次迭代中最小数值所在位置下标

        # 和其后所有位置上的数字比较，如果有更小的数字，则用该位置替代当前的minPosition
        for i in range(start_position+1, len(arr)):
            if (arr[i] <arr[min_position]):
                min_position = i

        # 经过一轮比较，当前的minPosition已经是当前待排序数字中的最小值，将它和本次迭代第一个位置上的数字交换
        swap(arr, start_position, min_position)
    return


# 下面代码为 14-2

arr = [3, 2, 1, 5, 8, 7, 9, 10, 13]
selection_sort(arr)
print(arr)