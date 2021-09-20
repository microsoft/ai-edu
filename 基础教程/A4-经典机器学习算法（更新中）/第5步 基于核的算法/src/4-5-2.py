
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def draw_left_A_L(ax):
    num_alpha = 17
    num_x = 17
    L = []
    # X 共 9 个等间距 x
    X = np.linspace(0,2,num_x)
    # A 共 9 个 a 
    A = np.linspace(-1,3,num_alpha)
   
    line1 = None
    # 固定 a，绘制所有可能的 x
    for a in A:
        l_xa = X*X - 2*X + 1 + a*(X-0.5)
        # 横坐标是 a, 纵坐标是函数值 L(x,a)，一条竖线
        line1, = ax.plot([a]*num_x, l_xa, marker='.', color='y', linestyle=':')
        L.append(l_xa)

    ax.legend(handles=[line1], labels=[u'在给定的a上尝试不同的x'])
    L = np.array(L)
    return L, A

def draw_right_A_L(ax):

    L, A = draw_left_A_L(ax)
    print(L)
    # D(a)
    min_x = np.min(L, axis=1)
    print("D(a) = min_x L(x,a) = ", min_x)
    ax.plot(A, min_x, label='$D(a) = min_x \ L(x,a)$', color='r', marker='o')

    # d* = max_a D(a)
    max_a = np.max(min_x)
    print("d* = max_a [min_x L(x,a)] = ", max_a)
    idx = np.argmax(min_x)
    print("a =", A[idx])
    ax.scatter(A[idx], max_a, s=50, color='r')
    ax.text(A[idx], max_a - 0.5, "d*", fontsize=16)
    ax.legend()


def dual_max_min():
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.title(u"对偶问题的极大极小值")
    plt.axis('off')

    ax = fig.add_subplot(121)
    ax.grid()
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('$L (x,\\alpha)$')
    draw_left_A_L(ax)

    ax2 = fig.add_subplot(122)
    ax2.grid()
    ax2.set_xlabel('$\\alpha$')
    ax2.set_ylabel('$L (x,\\alpha)$')
    draw_right_A_L(ax2)

    plt.show()

if __name__=="__main__":
    dual_max_min()
