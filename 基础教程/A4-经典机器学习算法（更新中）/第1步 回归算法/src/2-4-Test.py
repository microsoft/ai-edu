import numpy as np

def test_x_t_1():
    X = np.random.random(size=(10,3))
    result = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, X))
    print(np.round(result,2))

test_x_t_1()
