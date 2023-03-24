import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

num_arm = 3
num_data = 10

mpl.rcParams['font.sans-serif'] = ['SimHei']  
mpl.rcParams['axes.unicode_minus']=False


#正态分布的概率密度函数
def normpdf(x,mu,sigma):       
    pdf=np.exp(-(x-mu)**2/(2*sigma**2))/(sigma * np.sqrt(2 * np.pi))
    return pdf

def draw(x, y, color, y_label, title, label=None):
    #概率分布曲线
    plt.plot(x,y,color,linewidth=2,label=label)
    plt.title(title)
    #plt.xticks ([mu-2*sigma,mu-sigma,mu,mu+sigma,mu+2*sigma],['-2','-1','0','1','2'])
    plt.xlabel(u"奖励")
    plt.ylabel(y_label)


if __name__=="__main__":
    sigma = 1
    mu = 0
    bins = 30
    n = 200
    np.random.seed(5)
    a = sigma * np.random.randn(n) + mu
    plt.hist(a, bins=bins)

    x= np.arange(mu-4*sigma,mu+4*sigma,0.01) #生成数据，步长越小，曲线越平滑
    y=normpdf(x,mu,sigma) * (1/0.2*n/bins)
    title = '$\mu = {:.2f}, \sigma={:.2f}$'.format(mu,sigma)
    draw(x, y, 'r--', u"次数", title)
    plt.grid()
    plt.show()

    for mu, color, label in zip([-1,0,1],['b--','r--','g--'],['2','1','3']):
        x= np.arange(mu-3*sigma,mu+3*sigma,0.01) #生成数据，步长越小，曲线越平滑
        y=normpdf(x,mu,sigma)
        title = '$\mu = [-1,0,1], \sigma=1$'
        draw(x, y, color, u"概率密度", title, label)
    plt.legend()
    plt.grid()
    plt.show()
    
