import time

# 递归法
def f1(n):
    if n==1 or n==2:
        return 1
    else:
        fn = f1(n-1) + f1(n-2)
        return fn

# 迭代法
def f2(n):
    fn_2 , fn_1 = 1, 1
    for i in range(n-2):
        fn = fn_2 + fn_1
        fn_2 = fn_1
        fn_1 = fn
    return fn

# 备忘录法
def f3(n):
    results = {1:1, 2:2}    # 当n=1和n=2时的结果是1和2
    for i in range(3, n+1):
        fi = results[i-1] + results[i-2]    # 从字典中直接取出结果
        results[i] = fi     # 在字典中添加n=i时的结果,i=3,4,5...
    return results


if __name__=="__main__":
    start = time.time()
    print("递归法结果 =", f1(35))
    end1 = time.time()
    print("递归法耗时 =", end1-start)
    
    print("迭代法结果 = ", f2(35))
    end2 = time.time()
    print("迭代法耗时 =", end2-end1)

    print("备忘录法结果 =", f3(10))
    end3 = time.time()
    print("备忘录法耗时 =", end3-end2)
