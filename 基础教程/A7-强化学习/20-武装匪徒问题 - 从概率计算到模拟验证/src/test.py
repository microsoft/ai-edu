import numpy as np
import time

ss = np.random.randn(1000)

def aa():
    r = np.random.choice(ss)
    return r

def bb():
    r = np.random.rand()
    return r

import math

def softmax(a,b,c, t):
    at = math.exp(a/t)
    bt = math.exp(b/t)
    ct = math.exp(c/t)
    pa = at / (at+bt+ct)
    pb = bt / (at+bt+ct)
    pc = ct / (at+bt+ct)
    print(pa,pb,pc)

    at = math.exp((a-c)/t)
    bt = math.exp((b-c)/t)
    ct = math.exp((c-c)/t)
    pa = at / (at+bt+ct)
    pb = bt / (at+bt+ct)
    pc = ct / (at+bt+ct)
    print(pa,pb,pc)


if __name__=="__main__":

    softmax(1,2,3,0.5)
    softmax(1,2,3,0.8)
    softmax(1,2,3,1)
    softmax(1,2,3,2)
    exit(0)
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
