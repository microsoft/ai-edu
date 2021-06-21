def fibonacci(n):
    if n <0:
        print("Incorrect input")
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# 下面为 代码16-9

print(fibonacci(10))