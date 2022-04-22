
import math

def f(x):
    y = 10 + math.log10(x/2)
    return y

if __name__=="__main__":
    x = 100
    delta = 100
    count = 0
    while delta > 1e-5:     # 判断是否收敛
        count += 1          # 计数器
        y = f(x)            # 迭代计算
        delta = abs(x - y)  # 检查误差
        print(str.format("{0}: x={1}，相对误差={2}", count, y, delta))
        x = y               # 为下一次迭代做准备
