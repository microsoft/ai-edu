
import numpy as np

P = np.array([
    [0.6, 0.4],
    [0.1, 0.9]
])

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
    
    people = np.array([100000,20000])
    print("最初的农村-城市人口数量为：")
    print(people)
    final=np.dot(people, Pn)
    print("最终的农村-城市人口数量为：")
    print(np.around(final, 0))
