
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-100,100,100)
y = 4*x*x*x*x -50*x*x+100*x
y = 4*x*x*x -100*x+100
plt.plot(x,y)
plt.show()
