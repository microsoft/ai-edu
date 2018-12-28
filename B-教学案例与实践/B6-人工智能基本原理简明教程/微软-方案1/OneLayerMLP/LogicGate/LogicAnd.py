import numpy as np
import matplotlib.pyplot as plt

# x1=0,x2=0,y=0
# x1=0,x2=1,y=0
# x1=1,x2=0,y=0
# x1=1,x2=1,y=1
def ReadAndData():
    X = np.array([0,0,1,1,0,1,0,1]).reshape(2,4)
    Y = np.array([0,0,0,1]).reshape(1,4)
    return X,Y

def Sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

def ForwardCalculation(W,B,X):
    z = np.dot(W, X) + B
    a = Sigmoid(z)
    return a

def BackPropagation(X,Y,A):
    dloss_z = A - Y
    db = dloss_z
    dw = np.dot(dloss_z, X.T)
    return dw, db

def UpdateWeights(w, b, dW, dB, eta):
    w = w - eta*dW
    b = b - eta*dB
    return w,b

def CheckLoss(w, b, X, Y, count, prev_loss):
    A = ForwardCalculation(w, b, X)
    p1 = Y * np.log(A)
    p2 = (1-Y) * np.log(1-A)
    LOSS = -(p1 + p2)
    loss = np.sum(LOSS) / count
    diff_loss = abs(loss - prev_loss)
    return loss, diff_loss

def InitialParameters(num_input, num_output, flag):
    if flag == 0:
        # zero
        W1 = np.zeros((num_output, num_input))
    elif flag == 1:
        # normalize
        W1 = np.random.normal(size=(num_output, num_input))
    elif flag == 2:
        #
        W1=np.random.uniform(-np.sqrt(6)/np.sqrt(num_input+num_output),np.sqrt(6)/np.sqrt(num_output+num_input),size=(num_output,num_input))

    B1 = np.zeros((num_output, 1))
    return W1,B1

def ShowResult(W,B,X,Y):
    count = 100
    XF = np.zeros((2,0))
    XT = np.zeros((2,count))
    XT[0,:] = np.linspace(0,1,num=count)
    for i in range(count):
        prev_diff = 10
        y = np.linspace(0,1,num=count)
        for j in range(count):
            XT[1,i] = y[j]
            A2 = ForwardCalculation(W,B,XT[:,i].reshape(2,1))
            current_diff = np.abs(A2[0][0] - 0.5)
            if current_diff < 0.01:
                XF = np.append(XF, [[XT[0,i]],[XT[1,i]]], axis=1)
                break
            prev_diff = current_diff

    plt.axis([-0.1,1.2,-0.1,1.2])
    plt.plot(XF[0,:],XF[1,:],'g')

    for i in range(X.shape[1]):
        if Y[0,i] == 0:
            plt.scatter(X[0,i],X[1,i],marker="o",c='b',s=64)
        else:
            plt.scatter(X[0,i],X[1,i],marker="^",c='r',s=64)
    plt.show()
    return XF

def Test(W,B):
    a = input("input number one:")
    x1 = float(a)
    b = input("input number two:")
    x2 = float(b)
    z = ForwardCalculation(W, B, np.array([x1,x2]).reshape(2,1))
    print (z)

n_input = 2
n_output = 1
W,B = InitialParameters(n_input, n_output,1)

# initialize_data
eta = 0.1
eps = 1e-10
iteration, max_iteration = 0, 1000
LOSS = 0.01
# calculate loss to decide the stop condition
prev_loss, loss, diff_loss = 0,0,0
# create mock up data
X, Y = ReadAndData()
# count of samples
num_features = X.shape[0]
num_example = X.shape[1]

for iteration in range(max_iteration):
    for i in range(num_example):
        # get x and y value for one sample
        x = X[:,i].reshape(num_features,1)
        y = Y[:,i].reshape(1,1)
        # get z from x,y
        z = ForwardCalculation(W, B, x)
        # calculate gradient of w and b
        dW, dB = BackPropagation(x, y, z)
        # update w,b
        W, B = UpdateWeights(W, B, dW, dB, eta)
        # calculate loss for this batch
        loss, diff_loss = CheckLoss(W,B,X,Y,num_example,prev_loss)
        # condition 1 to stop
        if diff_loss < eps:
            break
        if loss < LOSS:
            break;
        prev_loss = loss
    print(iteration,i,loss,diff_loss,W,B)
    if diff_loss < eps:
        break
    if loss < LOSS:
        break;

print("w=",W)
print("b=",B)

ShowResult(W,B,X,Y)

# test
while True:
    Test(W,B)

