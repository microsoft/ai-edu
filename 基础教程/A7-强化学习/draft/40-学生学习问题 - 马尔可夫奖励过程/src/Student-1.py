import numpy as np

P = np.array(
    [   #Cl1  Cl2  Cl3  Pass Rest Game End
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0], # Class1
        [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2], # CLass2
        [0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0], # Class3
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], # Pass
        [0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0], # Rest
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0], # Game
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # End
    ]
)


def Check_Convergence(P):
    P_curr = P.copy()
    for i in range(100000):
        P_next=np.dot(P,P_curr)
        print("迭代次数 =",i+1)
        print(P_next)
        if np.allclose(P_curr, P_next):
            break
        P_curr = P_next
    return P_next

if __name__=="__main__":
    Pn = Check_Convergence(P)
