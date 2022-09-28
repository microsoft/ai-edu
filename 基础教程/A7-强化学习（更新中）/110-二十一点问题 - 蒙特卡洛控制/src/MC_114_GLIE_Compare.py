from cProfile import label
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus'] = False

def standard_GLIE(ax):
    A = []
    B = []
    for i in range(1000):
        epsilon = 1/(i+1)
        greedy = 1 - epsilon + epsilon / 4
        explor = 1 - greedy
        A.append(greedy)
        B.append(explor)
    ax.plot(A, label="贪心")
    ax.plot(B, label="探索")
    ax.grid()
    ax.legend()
    ax.set_title(u"传统的GLIE")

def improved_GLIE(ax):
    A = []
    B = []
    for i in range(1000):
        epsilon = 1 / (math.log(i+1,10)+1)
        greedy = 1 - epsilon + epsilon / 4
        explor = 1 - greedy
        A.append(greedy)
        B.append(explor)
    ax.plot(A, label="贪心")
    ax.plot(B, label="探索")
    ax.grid()
    ax.legend()
    ax.set_title(u"改进的GLIE")
    

if __name__=="__main__":
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121)
    standard_GLIE(ax1)
    ax2 = fig.add_subplot(122)
    improved_GLIE(ax2)
    plt.show()
