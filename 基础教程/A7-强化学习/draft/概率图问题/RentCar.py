import numpy as np

P = np.array([
    [0.5, 0.2, 0.3],
    [0.3, 0.1, 0.6],
    [0.3, 0.6, 0.1]
])

P1 = np.array([
    [0.1, 0.3, 0.0, 0.6],
    [0.8, 0.0, 0.2, 0.0],
    [0.0, 0.9, 0.1, 0.0],
    [0.0, 0.3, 0.3, 0.4]
])


X = np.array([0,1,0])
X1 = np.array([0,1,0,0])
X_curr = X.copy()
while True:
    X_next = np.dot(X_curr, P)
    print(X_next)
    if (np.allclose(X_curr, X_next)):
        break
    X_curr = X_next


for i in range(10):
    