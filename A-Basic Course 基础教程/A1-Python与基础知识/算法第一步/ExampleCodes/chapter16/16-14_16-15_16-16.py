from Utilities import qsort_recursion
from Utilities import generate_test_data

# 下面为 代码16-14
_, _, arr_reverse = generate_test_data(1, 1000)

# 下面一行为打印提示语句，不要出现在代码示例里：
print("\n\nFollowing are output of code 16-15\n")   # TOBE IGNORED

# 下面为 代码16-15
qsort_recursion(arr_reverse, 0, len(arr_reverse) - 1)
print("sorted :", arr_reverse)

# 下面一行为打印提示语句，不要出现在代码示例里：
print("\n\nFollowing are output of code 16-16\n")   # TOBE IGNORED

# 下面为 代码16-16
_, _, arr_reverse = generate_test_data(1, 2000)
qsort_recursion(arr_reverse, 0, len(arr_reverse) - 1)
print("sorted arrReverse:",arr_reverse)