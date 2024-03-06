
def cal_v(v, alpha, gamma, reward, v_next):
    return v + alpha * (reward + gamma * v_next - v)

if __name__ == "__main__":
    v = 0.5
    alpha = 0.1
    gamma = 0.9
    reward = 1
    v_next = 2

    v_old = v
    for i in range(1000):
        v = cal_v(v, alpha, gamma, reward, v_next)
        print(i,v)
        if abs(v - v_old) < 1e-4:
            break
        v_old = v

