import numpy as np
import matplotlib.pyplot as plt

num_arm = 10
num_data = 200

a = np.random.randn(num_arm)
b = np.random.randn(num_data, num_arm)

print("a=", a)
print("b=", b)
c = b + a
print(c.shape)
print("c=", b+a)

plt.hist(a, bins=5)
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=4)
for i in range(4):
    ax = axes[0,i]
    ax.grid()
    ax.hist(b[:,i])

for i in range(4):
    ax = axes[1,i]
    ax.grid()
    ax.hist(c[:,i])

plt.show()

plt.violinplot(c, showmeans=True)
plt.grid()
plt.show()
