import numpy as np
import operator as op

# 作为目标的转移概率矩阵
P = np.array([
    [0.1, 0.3, 0.0, 0.6],
    [0.8, 0.0, 0.2, 0.0],
    [0.0, 0.9, 0.1, 0.0],
    [0.0, 0.3, 0.3, 0.4]
])

# 采样
def sample(n_samples, n_states, start_state):
    states = [i for i in range(n_states)]
    # 状态转移序列
    X = []
    # 开始采样
    X.append(start_state)
    current = start_state
    for i in range(n_samples):
        next = np.random.choice(states, p=P[current])
        X.append(next)
        current = next
    #endfor
    return X

# 生成P'
def cal_P(X):
    count = {}
    for i in X[0:len(X) - 1]:
        count[i] = count.get(i, 0) + 1
    count = sorted(count.items(), key=op.itemgetter(0), reverse=False)
    print(count)
    
    P1 = np.zeros([len(count), len(count)])
    for j in range(len(X) - 1):
        for m in range(len(count)):
            for n in range(len(count)):
                if X[j] == count[m][0] and X[j + 1] == count[n][0]:
                    P1[m][n] += 1
    for t in range(len(count)):
        P1[t, :] /= count[t][1]
    return P1


def cal_P1(n_states, X):
    P1 = np.zeros((n_states, n_states))
    for i in range(len(X)-1):
        x = X[i]
        next_x = X[i+1]
        P1[x, next_x] += 1
    #endfor
    a = np.sum(P1, axis=1, keepdims=True)
    print(a)
    b = P1 / a
    return b

if __name__ == "__main__":
    # 采样数量
    n_samples = 100000
    # 状态空间
    n_states = 4
    # 起始状态(从0开始)
    start_state = 1
    X = sample(n_samples, n_states, start_state)
    # 转化成A,B,C,D
    #print(X)
    #Y = [chr(x+65) for x in X]
    #print(Y)

    P1 = cal_P(X)
    print(P1)

    print("----------")

    P2 = cal_P1(n_states, X)
    print(P2)



    
