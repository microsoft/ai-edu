import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

num_arm = 3
num_data = 10

mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus']=False


sigma = 1
mu = 0
bins = 30
n = 200
np.random.seed(5)
a = sigma * np.random.randn(n) + mu
plt.hist(a, bins=bins)


#正态分布的概率密度函数
def normpdf(x,mu,sigma):       
    pdf=np.exp(-(x-mu)**2/(2*sigma**2))/(sigma * np.sqrt(2 * np.pi))
    return pdf

x= np.arange(mu-3*sigma,mu+3*sigma,0.01) #生成数据，步长越小，曲线越平滑
y=normpdf(x,mu,sigma) * (1/0.2*n/bins)

#概率分布曲线
plt.plot(x,y,'r--',linewidth=2)
plt.title('$\mu = {:.2f}, \sigma={:.2f}$'.format(mu,sigma))
#plt.vlines(mu, 0, normpdf(mu,mu,sigma), colors = "c", linestyles = "dotted")
#plt.vlines(mu+sigma, 0, normpdf(mu+sigma,mu,sigma), colors = "y", linestyles = "dotted")
#plt.vlines(mu-sigma, 0, normpdf(mu-sigma,mu,sigma), colors = "y", linestyles = "dotted")
plt.xticks ([mu-2*sigma,mu-sigma,mu,mu+sigma,mu+2*sigma],['-2','-1','0','1','2'])
plt.xlabel(u"奖励")
plt.ylabel(u"次数")

#输出
plt.grid()
plt.show()

exit(0)

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
