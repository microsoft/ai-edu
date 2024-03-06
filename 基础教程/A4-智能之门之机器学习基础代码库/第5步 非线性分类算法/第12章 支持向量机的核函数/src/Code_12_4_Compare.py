import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs,make_circles,make_moons,make_classification
from sklearn.svm import SVC
from sklearn.preprocessing import *

#生成数据
n_samples = 500
datasets = [
    make_blobs(n_samples = n_samples,centers = 2,cluster_std=0.2,random_state = 0)
    ,make_circles(n_samples = n_samples,noise = 0.2,factor = 0.4,random_state=0)
    ,make_moons(n_samples = n_samples,noise = 0.2,random_state=0)
    ,make_classification(n_samples = n_samples,n_features=2,n_informative=2,n_redundant=0,random_state=20)
]
#make_classification参数:
#n_informative=, # 有效特征个数
#n_redundant=, # 冗余特征个数（有效特征的随机组合）
#n_repeated=, # 重复特征个数（有效特征和冗余特征的随机组合）
#n_classes=, # 样本类别
#n_clusters_per_class=, # 簇的个数

#创建一行四列子图
fig,axes = plt.subplots(nrows=1,ncols=4,figsize=(20,4))
for i,(X,Y) in enumerate(datasets):

    ax = axes[i]
    ax.scatter(X[:,0],X[:,1],cmap='rainbow',c=Y)
    ax.set_xticks([])
    ax.set_yticks([])

kernels = ['linear','poly','sigmoid','rbf']
nrows = len(datasets)
ncols = len(kernels)+1

fig,axes = plt.subplots(nrows = nrows,ncols = ncols,figsize=(20,16))

for i,(X_raw,y) in enumerate(datasets):

    ss = StandardScaler()
    X = ss.fit_transform(X_raw)


    ax = axes[i,0]
    if i == 0:
        ax.set_title('Input Data')
    
    ax.scatter(X[:,0],X[:,1],cmap = 'rainbow',c = y,s=3)
    ax.set_xticks([])
    ax.set_yticks([])
    
    for j in range(1,ncols):
        ax = axes[i,j]
        if i == 0 :
            ax.set_title('kernel = {}'.format(kernels[j-1]))
        ax.scatter(X[:,0],X[:,1],cmap = 'rainbow',c=y,s=3)
        clf = SVC(kernel = kernels[j-1]).fit(X,y)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        axisx = np.linspace(xlim[0],xlim[1],300)
        axisy = np.linspace(ylim[0],ylim[1],300)
        axisx,axisy = np.meshgrid(axisx,axisy)
        xy = np.vstack([axisx.ravel(),axisy.ravel()]).T
        
        Z = clf.decision_function(xy).reshape(axisx.shape)
        
#        ax.contour(axisx,axisy,Z
#                  ,colors = 'k'
#                  ,linestyles = ['--','-','--']
#                  ,levels = [-1,0,1]
#                  )
        ax.contourf(axisx,axisy,Z)

        #填充等高线不同区域的颜色
        #ax.pcolormesh(axisx, axisy, Z > 0, cmap=plt.cm.Paired)
        plt.text(0.95,0.06#坐标轴的相对位置
                 ,'{:.2f}'.format(clf.score(X,y)*100)
                ,bbox = {'boxstyle':'round','alpha':0.8,'facecolor':'white'}
                ,transform=ax.transAxes #确定文字所对应的坐标轴，就是ax子图的坐标轴本身
                ,horizontalalignment='right' #位于坐标轴的什么方向
                )

plt.show()
