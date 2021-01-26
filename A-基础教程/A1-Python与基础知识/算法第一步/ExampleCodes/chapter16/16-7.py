def recursion_test(depth):
    print("recursion depth:", depth)
    if depth < 996:
        recursion_test(depth + 1)
    else:
        print("exit recursion")
    return

# 下面是 代码16-5，重复出现，不要出现在代码示例里：

recursion_test(1)