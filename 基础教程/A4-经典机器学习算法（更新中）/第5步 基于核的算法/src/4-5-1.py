
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def draw_left_X_L(ax, flag='left'):
    num_alpha = 5
    num_x = 41
    L = []
    # X 共 41 个等间距 x
    X = np.linspace(-1,3,num_x)
    # A=0,0.5,1,1.5,2 共 5 个选择
    A = np.linspace(0,2,num_alpha)
   
    line1 = None
    # 固定 x，绘制所有可能的 alpha
    for x in X:
        l_xa = x*x - 2*x + 1 + A*(x-0.5)
        # 横坐标是 x, 纵坐标是函数值 L(x,a)，一条竖线
        line1, = ax.plot([x]*5, l_xa, color='y', marker='.', linestyle=':')
        L.append(l_xa)

    if (flag == 'left'):
        # 绘制原函数 f(x)
        Y = X**2 - 2*X + 1
        line2, = ax.plot(X, Y, color='r')
        ax.legend(handles=[line1, line2], labels=[u'在给定的x上尝试不同的a', '原函数 f(x)'])

    L = np.array(L)
    return L, X

def draw_right_X_L(ax):
    L, X = draw_left_X_L(ax, 'right')

    print(L)
    # P(x) = max_a L(x,a)
    max_a = np.max(L, axis=1)
    print("max_a L(x,a) =", max_a)
    # d* = min_x P(x)
    min_x = np.min(max_a)
    print("p* = min_x [max_a L(x,a)] =", min_x)
    
    idx = np.argmin(max_a)
    print("x =",X[idx])
    # x <= 0.5
    ax.plot(X[0:15], max_a[0:15], color='r', marker='o', label='$P(x)=max_a \ L(x,a), x <= 0.5$')
    # x > 0.5
    ax.plot(X[15:], max_a[15:], color='r', linestyle=':', label='$P(x)=max_a \ L(x,a), x > 0.5$')
    ax.scatter(X[idx], min_x, s=50, color='r')
    ax.text(X[idx], min_x + 0.2, 'p*', fontsize=16)
    ax.legend()

    return L, X


def primal_min_max():
    mpl.rcParams['font.sans-serif'] = ['SimHei']  
    mpl.rcParams['axes.unicode_minus']=False
    fig = plt.figure()
    plt.title(u"原始问题的极小极大值")
    plt.axis('off')

    ax = fig.add_subplot(121)
    ax.grid()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$L (x,\\alpha)$')
    draw_left_X_L(ax)
    
    ax2 = fig.add_subplot(122)
    ax2.grid()
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$L (x,\\alpha)$')
    draw_right_X_L(ax2)
       
    plt.show()


if __name__=="__main__":
    #min_max()
    primal_min_max()
