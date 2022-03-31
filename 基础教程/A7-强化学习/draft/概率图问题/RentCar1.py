import numpy as np

P = np.array([
    [0.1, 0.3, 0.0, 0.6],
    [0.8, 0.0, 0.2, 0.0],
    [0.0, 0.9, 0.1, 0.0],
    [0.0, 0.3, 0.3, 0.4]
])

def calculate_day(X, P, day):
    X_curr = X.copy()
    for i in range(day):
        print(str.format("day {0}: {1} ", i, X_curr))
        X_next = np.dot(X_curr, P)
        X_curr = X_next.copy()
        
def n_step_tran(P, n):
    P0=P.copy()
    for i in range(n-2):
        P1=np.dot(P,P0)
        P0 = P1.copy()
    print(P1)
    return P1

def tran(P):
    P0=P.copy()
    for i in range(100):
        P1=np.dot(P0,P)
        if np.allclose(P1, P0):
            break
        P0 = P1.copy()
    print(i)
    print(P1)


def aaa():
    X = np.array([1,0,0,0])
    X_curr = X.copy()
    while True:
        X_next = np.dot(X_curr, P)
        print(X_next)
        if (np.allclose(X_curr, X_next)):
            break
        X_curr = X_next

if __name__=="__main__":
    X = np.array([0,1,0,0])
    calculate_day(X, P, 10)
    P3 = n_step_tran(P, 10)
    print(np.dot(X, P3))

    #tran(P)