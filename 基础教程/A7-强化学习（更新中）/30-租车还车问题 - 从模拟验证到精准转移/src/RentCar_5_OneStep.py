import numpy as np

P = np.array([
    [0.1, 0.3, 0.0, 0.6],
    [0.8, 0.0, 0.2, 0.0],
    [0.0, 0.9, 0.1, 0.0],
    [0.0, 0.3, 0.3, 0.4]
])

def calculate_day(X, P, day):
    X_n = X.copy()
    for i in range(day+1): # 因为是从0开始，所以day要+1
        print(str.format("day {0}: {1} ", i, X_n))
        X_n = np.dot(X_n, P)

if __name__=="__main__":
    X = np.array([0,1,0,0]) # 该车第0天在B店
    calculate_day(X, P, 5)  # 计算第5天在哪里
