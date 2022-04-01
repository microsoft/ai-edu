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

if __name__=="__main__":
    X = np.array([0,1,0,0])
    calculate_day(X, P, 6)
