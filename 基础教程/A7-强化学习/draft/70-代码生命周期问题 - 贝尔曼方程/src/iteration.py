
import math

def fun(x):
    y = 10 + math.log10(x/2)
    return y

if __name__=="__main__":
    x = 100
    delta = 100
    while delta > 1e-5:
        y = fun(x)
        delta = abs(x - y)
        print(y, delta)
        x = y
