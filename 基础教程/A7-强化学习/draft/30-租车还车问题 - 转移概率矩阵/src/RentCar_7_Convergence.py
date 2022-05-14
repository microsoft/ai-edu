import numpy as np

P = np.array([
    [0.1, 0.3, 0.0, 0.6],
    [0.8, 0.0, 0.2, 0.0],
    [0.0, 0.9, 0.1, 0.0],
    [0.0, 0.3, 0.3, 0.4]
])

def Check_Convergence(P):
    P_curr = P.copy()
    for i in range(1000):
        P_next=np.dot(P,P_curr)
        print("迭代次数 =",i+1)
        print(P_next)
        if np.allclose(P_curr, P_next):
            break
        P_curr = P_next
    return P_next

if __name__=="__main__":
    Pn = Check_Convergence(P)
