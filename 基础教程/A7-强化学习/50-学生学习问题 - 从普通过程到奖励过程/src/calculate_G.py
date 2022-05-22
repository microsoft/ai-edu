
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
    calcualte_G([0,1,1,0,1,0,-1,4], 1)
    calcualte_G([0,1,1,0,1,0,-1,4], 0.5)
    calcualte_G([0,1,1,1,0,4], 1)
    calcualte_G([0,1,1,1,0,4], 0.5)
    calcualte_G([0,1,-3,0,1,0,-1,4], 1)
    calcualte_G([0,1,-3,0,1,0,-1,4], 0.5)
    calcualte_G([0,1,1,1,-2,0,1,-1,-4], 1)
    calcualte_G([0,1,1,1,-2,0,1,-1,-4], 0.5)