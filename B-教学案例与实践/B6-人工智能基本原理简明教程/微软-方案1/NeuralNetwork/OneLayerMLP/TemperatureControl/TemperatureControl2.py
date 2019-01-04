import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

x_data_name = "TemperatureControlXData.npy"
y_data_name = "TemperatureControlYData.npy"

def ReadData():
    Xfile = Path(x_data_name)
    Yfile = Path(y_data_name)
    if Xfile.exists() & Yfile.exists():
        X = np.load(Xfile)
        Y = np.load(Yfile)
        return X,Y
    else:
        return None,None

def ForwardCalculation(w,b,x):
    z = w * x + b
    return z

def BackPropagationBatch(X, Y, Z, count):
    dZ = Z - Y
    dB = sum(dZ)/count
    q = dZ * X
    dW = sum(q)/count
    return dW, dB

def BackPropagationSingle(x,y,z):
    dZ = z - y
    dB = dZ
    dW = dZ * x
    return dW, dB

def UpdateWeights(w, b, dW, dB, eta):
    w = w - eta*dW
    b = b - eta*dB
    return w,b

def check_diff(w, b, X, Y, count, prev_loss):
    Z = w * X + b
    LOSS = (Z - Y)**2
    loss = LOSS.sum()/count/2
    diff_loss = abs(loss - prev_loss)
    return loss, diff_loss

def show_result(X, Y, w, b, iteration, loss_his, w_his, b_his, n):
    # draw sample data
    #plt.subplot(121)
    #plt.plot(X, Y, "b.")
    # draw predication data
    Z = w*X +b
 #   plt.plot(X, Z, "r")
 #   plt.subplot(122)
    plt.plot(loss_his[100:n], "r")
    #plt.plot(w_his[0:n], "b")
    #plt.plot(b_his[0:n], "g")
#    plt.grid(True)
    plt.show()
    print(iteration)
    print(w,b)

def print_progress(iteration, loss, diff_loss, w, b, loss_his, w_his, b_his):
    #if iteration % 10 == 0:
    print(iteration, loss, diff_loss, w, b)
    loss_his = np.append(loss_his, loss)
    w_his = np.append(w_his, w)
    b_his = np.append(b_his, b)
    return loss_his, w_his, b_his



# initialize_data
eta = 0.1
# set w,b=0, you can set to others values to have a try
w, b = 0, 0
eps = 1e-10
iteration, max_iteration = 0, 100
# calculate loss to decide the stop condition
prev_loss, loss, diff_loss = 0,0,0
# create mock up data
X, Y = ReadData()
# count of samples
m = X.shape[0]
loss_his = list()
w_his = list()
b_his = list()

for iteration in range(max_iteration):
    for i in range(m):
        # get x and y value for one sample
        x = X[i]
        y = Y[i]
        # get z from x,y
        z = ForwardCalculation(w, b, x)
        # calculate gradient of w and b
        dW, dB = BackPropagationSingle(x, y, z)
        # update w,b
        w, b = UpdateWeights(w, b, dW, dB, eta)
        # calculate loss for this batch
        loss, diff_loss = check_diff(w,b,X,Y,m,prev_loss)
        # condition 1 to stop
        loss_his, w_his, b_his = print_progress(iteration, loss, diff_loss, w, b, loss_his, w_his, b_his)        
        if diff_loss < eps:
            break
        prev_loss = loss

    if diff_loss < eps:
        break

show_result(X, Y, w, b, iteration, loss_his, w_his, b_his, 200)
