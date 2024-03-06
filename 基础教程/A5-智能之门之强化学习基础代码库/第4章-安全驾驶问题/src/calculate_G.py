
import math

def calcualte_G(R, gamma):
    T = len(R)
    G = 0
    for t in range(T):
        discount = math.pow(gamma, t)
        G += discount * R[t]
    print("奖励序列 =", R)
    print("gamma =",gamma)
    print("G =",G)
    print("-----")
    return G


if __name__=="__main__":
    calcualte_G([0,0,1,5], 1)
    calcualte_G([0,0,1,0,1,5], 1)
    calcualte_G([0,0,-6,-1], 1)
    calcualte_G([0,-3,-6,0,-6,-1], 1)
    calcualte_G([0,0,1,5], 0.9)
    calcualte_G([0,0,1,0,1,5], 0.9)
    calcualte_G([0,0,-6,-1], 0.9)
    calcualte_G([0,-3,-6,0,-6,-1], 0.9)
