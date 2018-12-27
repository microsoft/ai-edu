# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt


def TargetFunction(x):
    y = np.sin(x)*1.2+x*0.3
    return y

a = np.random.randint(0,1000,100)/100
b = TargetFunction(a)


x1 = np.random.randint(0,1000,200)/100
x2 = np.random.randint(0,30,200)/10
y = np.zeros(200)
for i in range(x2.shape[0]):
    t = TargetFunction(x1[i])
    if x2[i] > t:
        y[i] = 0
        plt.scatter(x1[i],x2[i], c='r')
    else:
        y[i] = 1
        plt.scatter(x1[i],x2[i], c='b')



plt.scatter(a,b)
plt.xlabel("Years")
plt.ylabel("Seasons")
plt.title("Worm Probability")
plt.show()
X = np.zeros((2,200))
X[0,:]=x1
X[1,:]=x2
print(X)
print(y)

np.save("TreeWormXData.dat", X)
np.save("TreeWormYData.dat", y)

