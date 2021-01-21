import random


def generate_test_data(start, end, len=None):
    arr_random = None

    if len is not None:
        arr_random = [random.randint(start, end) for x in range(0, len)]
    arr_seq = [x for x in range(start, end + 1)]
    arr_reverse = [end + 1 - x for x in range(start, end + 1)]

    return arr_random, arr_seq, arr_reverse


# 下面为16-10
arr_random, arr_seq, arr_reverse = generate_test_data(1, 10, 5)
print(arr_random)
print(arr_seq)
print(arr_reverse)
