
import math

def f(x):
    y = 10 + math.log10(x/2)
    return y

if __name__=="__main__":
    x = 100
    delta = 100
    count = 0
    while delta > 1e-5:
        count += 1
        y = f(x)
        delta = abs(x - y)
        print(str.format("{0}: 结果={1}，误差={2}", count, y, delta))
        x = y
