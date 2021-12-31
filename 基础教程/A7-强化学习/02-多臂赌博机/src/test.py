import numpy as np
import time

ss = np.random.randn(1000)

def aa():
    r = np.random.choice(ss)
    return r

def bb():
    r = np.random.rand()
    return r

if __name__=="__main__":

    np.random.choice(ss)

    s = time.time()
    for run in range(1000):
        for step in range(1000):
            aa()

    e1 = time.time()

    for run in range(1000):
        for step in range(1000):
            bb()

    e2 = time.time()
    print(e1-s, e2-e1)
