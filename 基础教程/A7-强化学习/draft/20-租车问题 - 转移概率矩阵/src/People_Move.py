
import numpy as np

P = np.array([
    [0.7, 0.3],
    [0.1, 0.9]
])

def Check_Convergence(P, People_orginal):
    P_next = P.copy()
    People_next = People_orginal.copy()
    for i in range(1000000):
        People_curr = People_next.copy()
        print("---- 迭代次数 =",i+1)
        People_next = np.round(np.dot(People_orginal, P_next), 0)
        print("当前人口数量（农村,城市）", People_next)
        P_next = np.dot(P, P_next)
        print(P_next)
        if np.allclose(People_next, People_curr):
            break
    return P_next

if __name__=="__main__":
    people = np.array([100000,20000])
    print("最初的农村-城市人口数量为：")
    print(people)
    Pn = Check_Convergence(P, people)
    print("最终的农村-城市人口数量为：", np.around(np.dot(people, Pn),0))
    print("收敛的转移概率：")
    print(np.round(Pn,2))
