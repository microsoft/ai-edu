
import numpy as np
import matplotlib.pyplot as plt

def create_data(count):
    x1 = np.linspace(0,3.14,count)
    noise = (np.random.random(count)-0.5)/4
    y1 = np.sin(x1) + noise
    plt.scatter(x1,y1)

    x2 = np.linspace(0,3.14,count)
    noise = (np.random.random(count)-0.5)/4
    y2 = np.sin(x2) + noise + 0.65
    plt.scatter(x2, y2)

    plt.show()

if __name__ == '__main__':
    create_data(50)