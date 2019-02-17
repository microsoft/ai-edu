
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

x_data_name = "TemperatureControlXData.dat"
y_data_name = "TemperatureControlYData.dat"

def ReadData():
    Xfile = Path(x_data_name)
    Yfile = Path(y_data_name)
    if Xfile.exists() & Yfile.exists():
        X = np.load(Xfile)
        Y = np.load(Yfile)
        return X,Y
    else:
        return None,None

max_iteration=1
eta = 0.1
X, Y = ReadData()
w, b = 0.0, 0.0
# count of samples
num_example = X.shape[0]
for iteration in range(max_iteration):
    for i in range(num_example):
        # get x and y value for one sample
        x = X[i]
        y = Y[i]
        # get z from x,y
        z = w*x+b
        # calculate gradient of w and b
        dz = z - y
        db = dz
        dw = dz * x
        # update w,b
        w = w - eta * dw
        b = b - eta * db

print(w,b)