import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0,1,100)
y1=-np.log(x)
plt.plot(x,y1)
plt.title("y=1, Loss=-lna")
plt.xlabel("predication")
plt.ylabel("Loss")
plt.show()

y2=-np.log(1-x)
plt.plot(x,y2)
plt.title("y=0, Loss=-ln(1-a)")
plt.xlabel("predication")
plt.ylabel("Loss")
plt.show()
