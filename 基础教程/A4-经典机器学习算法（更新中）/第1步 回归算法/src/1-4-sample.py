import numpy as np

# 公式 1.2.5
def least_square_1(X,Y):
    n = X.shape[0]
    # a_hat
    numerator = n * np.sum(X*Y) - np.sum(X) * np.sum(Y)
    denominator = n * np.sum(X*X) - np.sum(X) * np.sum(X)
    a_hat = numerator / denominator
    # b_hat
    b_hat = (np.sum(Y - a_hat * X))/n
    return a_hat, b_hat

def mse(Y, Y_hat):
    loss = np.sum((Y-Y_hat) * (Y-Y_hat))
    return loss

if __name__ == '__main__':
    X = np.array([2,3,4]).reshape(3,1)
    Y = np.array([2,3,3]).reshape(3,1)
    a,b=least_square_1(X,Y)
    print(a,b)
    Y_hat = a * X + b
    loss = mse(Y, Y_hat)
    print(loss)
