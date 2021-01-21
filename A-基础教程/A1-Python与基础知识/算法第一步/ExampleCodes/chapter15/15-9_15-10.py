from Utilities import swap


def partition_v2_print(arr, low, high):
    if low >= high:
        return -1

    pi = low
    li = low + 1
    ri = high

    print("original list: ", arr)

    while ri >= li:
        print("\n[in loop] -- pi: ", pi, "li: ", li, "ri: ", ri)
        if arr[li] > arr[pi]:
            swap(arr, ri, li)
            print("\nswapped list: ", arr)
            ri -= 1
        else:
            li += 1

    print("\n[out of loop] -- arr: ", arr)
    print("[out of loop] -- pi: ", pi, "li: ", li, "ri: ", ri)

    pi = li - 1
    swap(arr, low, pi)

    print("\n[final] arr:", arr)
    print("[final] pi: ", pi)

    return pi


# 下面是 代码15-10

arr = [19, 11, 27, 8]
partition_v2_print(arr, 0, len(arr) - 1)
