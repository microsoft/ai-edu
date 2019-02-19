import numpy as np
import matplotlib.pyplot as plt

def draw_fun():
    x = np.linspace(1,10)
    a = x*x
    b = np.log(a)
    c = np.sqrt(b)
    plt.plot(x,c)
    #plt.show()

    d = 1/(x*np.sqrt(np.log(x)))
    plt.plot(x,d)
    plt.show()


if __name__ == '__main__':
    print("how to play: 1) input x, 2) calculate c, 3) input target number but not faraway from c")
    print("input x as initial number:")
    line = input()
    x = float(line)
    a = x*x
    b = np.log(a)
    c = np.sqrt(b)
    print("c=%f" %c)
    print("input y as target number(not farway from c):")
    line = input()
    y = float(line)
    while(True):
        print("forward...")
        # forward
        a = x*x
        b = np.log(a)
        c = np.sqrt(b)
        print(x,a,b,c)
        print("backward...")
        # backward
        loss = c - y
        if np.abs(loss) < 0.001:
            break
        delta_c = loss
        delta_b = delta_c * 2 * np.sqrt(b)
        delta_a = delta_b * a
        delta_x = delta_a / 2 / x
        x = x - delta_x
        print(delta_c, delta_b, delta_a, delta_x)

    print(x,a,b,c,loss)
    draw_fun()



