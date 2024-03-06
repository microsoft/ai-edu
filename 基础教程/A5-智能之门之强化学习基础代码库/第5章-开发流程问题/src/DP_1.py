import time
import common_helper as helper

# 递归法
def f1(n):
    if n==1 or n==2:
        return n
    else:
        fn = f1(n-1) + f1(n-2)  # 递归调用函数本身两次 f(n)=f(n-1)+f(n-2)
        return fn

# 迭代法
def f2(n):
    fn_2 , fn_1 = 1, 2      # f(n-2)=1, f(n-1)=2
    for i in range(n-2):
        fn = fn_2 + fn_1    # 计算 f(n) = f(n-2)+f(n-1)
        fn_2 = fn_1         # 迭代更换上一次的数值便于下次计算
        fn_1 = fn
    return fn

# 备忘录法
def f3(n):
    results = {1:1, 2:2}    # 建立字典，初始时，当n=1和n=2时的结果是1和2
    for i in range(3, n+1):
        fi = results[i-1] + results[i-2]    # 从字典中直接取出结果计算f(n)=f(n-2)+f(n-1)
        results[i] = fi     # 在字典中添加n=i时的结果,i=3,4,5...
    return results


if __name__=="__main__":
    for n in (6, 40):
        helper.print_seperator_line(helper.SeperatorLines.long)
        start = time.time()
        print("递归法结果 =", f1(n))
        end1 = time.time()
        print("递归法耗时 =", end1-start)
        helper.print_seperator_line(helper.SeperatorLines.long)
        print("迭代法结果 = ", f2(n))
        end2 = time.time()
        print("迭代法耗时 =", end2-end1)
        helper.print_seperator_line(helper.SeperatorLines.long)
        print("备忘录法结果 =", f3(n))
        end3 = time.time()
        print("备忘录法耗时 =", end3-end2)