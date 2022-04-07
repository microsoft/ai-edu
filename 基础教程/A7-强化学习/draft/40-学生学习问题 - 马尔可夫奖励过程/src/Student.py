import numpy as np

P = np.array(
    [   #Game Cl1  Cl2  Cl3  Pass Rest End
        [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], 
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.2],
        [0.0, 0.0, 0.0, 0.0, 0.6, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.2, 0.4, 0.4, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] 
    ]
)

def Check_Convergence(P):
    P_curr = P.copy()
    for i in range(100000):
        P_next=np.dot(P,P_curr)
        print("迭代次数 =",i+1)
        print(np.around(P_next, 2))
        if np.allclose(P_curr, P_next, rtol=1e-2, atol=1e-4):
            break
        P_curr = P_next
    return P_next

if __name__=="__main__":
    Pn = Check_Convergence(P)
