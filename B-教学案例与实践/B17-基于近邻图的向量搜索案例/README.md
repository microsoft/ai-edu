Copyright © Microsoft Corporation. All rights reserved.
  适用于[License](https://github.com/Microsoft/ai-edu/blob/master/LICENSE.md)版权许可
  

# 基于近邻图的向量搜索
约一年前，微软开源了一种基于近邻图的向量搜索算法[SPTAG](https://github.com/microsoft/SPTAG)，该算法能够在大规模向量中快速搜索最近邻点，被用于[Microsoft Bing](https://bing.com)中。我们将在此案例中介绍该搜索算法。

# 案例介绍
## 使用场景
如今，随着信息的逐渐增多以及用户习惯的改变，网络搜索变得更加困难。工程师们发现用户搜索越来越长，他们往往期望通过输入一段问题甚至一张图片来搜索出他们想要的内容，而不再是通过几个关键词。

基于此，搜索引擎开始利用向量来增强搜索效果。向量可以是关键词、图片等内容在高维空间的表示，我们可以通过已有的算法（如深度神经网络等）将关键词、图片等映射到高维空间，获取到对应的向量。向量在高维空间的表示是有意义的，我们认为关联性越强的内容在空间上的距离越小或相似度越大。利用向量，我们可以更轻易地找出相关联的内容，同时，这也让“以图搜图”成了可能，如Bing的图片搜索。

![Bing的图片搜索](./resource/bingsearch.png)

传统的关键词检索往往通过构建倒排索引（Inverted Index）来提高检索速度，但是我们无法为向量构建这样的索引，通常是利用树或图（如KD-Tree, BKTree, KNN Graph）来构建索引，从而帮助我们在大量的向量中找出最相似的向量。

但使用树或图都有各自的问题，因此微软提出了SPTAG来快速搜索向量。


# 核心知识点
* KD-Tree (K-Dimensional Tree)
* BKTree (Balanced K-Means Tree)
* KNN Graph (K-Nearest Neighbor Graph)
* RNG (Relative Neighborhood Graph)

# 先修知识
* C++
* 数据结构（包含二叉树、哈希算法、图论等基础知识）


# 推荐学习时长
该案例推荐学习时长为：1.5小时

# 案例详解

## 问题定义
SPTAG解决的问题是如何从大规模的向量中快速找出最近邻点（Nearest Neighbor）。

我们可以将问题定义为：

![](./resource/defineProblem.png)

其中，q为查询向量，x为样本向量，我们可以计算它们的L2或余弦距离，获得两者距离最近的样本。

要实现这个目标，有几种常见的最近邻搜索算法（Nearest Neighbor Search）:
1. 基于哈希的最近邻搜索
   
   利用哈希算法（如LSH），在尽可能保留距离关系的情况下，将样本映射到不同的哈希桶（Bucket）中，这时只需比较同一哈希桶中的点即可。但是该方法的查询性能与哈希函数及样本分布有关，样本可能会聚集在某些哈希桶中，导致对于不同点的查询时延差距较大，稳定性不佳。

2. 空间划分树
   
   空间划分树常用的是KD-Tree，通过递归地选取K维作为结点划分依据，将样本划分成左子树和右子树，最终生成一棵二叉树索引。这类方法通常对于低维度的数据效果比较好（如小于100维），但对于高维数据效果较差。而图片的向量表示通常能够达到1000维甚至更多。

3. 近邻图
   
    通过使样本中所有点连接其近邻点，我们可以构建一张近邻图。在搜索时我们可以快速找到与查询点相连的近邻点。但是，我们无法确保我们构建的近邻图是连通图，因此有可能会陷入局部最优。



## SPTAG架构

可见，上述提到的算法都有各自的问题，适用于不同的场景，而SPTAG的核心思路是将树和图结合，从而弥补各自的缺陷，使场景更为通用。

其架构如图：

![](./resource/tree+graph.jpg)

SPTAG分为了Tree部分和Graph部分。Tree部分利用KD-Tree或BKTree实现，Graph部分使用了基于KNN图改进的KNG。在进行搜索时，SPTAG首先会从Tree部分获取“种子”向量，将该种子向量作为Graph中的起始点进一步搜索近邻点。

## Tree部分
Tree部分SPTAG使用了KDTree和BKTree实现。在调用时，可以根据需求选择任意一种。KD-Tree适合低维度的向量，反之，BKTree适合高维度的向量。

### KD-Tree


#### 构建

1. 从方差最大的前5个维度中随机选择一个维度作为划分维度，将其平均值作为划分值，划分出两组子空间
2. 分别对划分的子空间递归以上步骤，直到划分的子空间中只有一个点，然后将其作为叶子结点

详细构建算法可以参考：[KD Tree的原理及Python实现](https://zhuanlan.zhihu.com/p/45346117)

#### 搜索
1. 选择到达叶子结点的路径中的最近邻点，将该点的向量作为后续在图中搜索的“种子”向量。


### BKTree

#### 构建

1. 每次使用平衡Kmeans聚类划分K组子空间
2. 分别对划分的子空间递归以上步骤，直到无法继续划分（所有结点都相同或子空间太小），将叶子结点指向这些数据点。

#### 搜索
1. 使用Best-Frist Search的方式搜索BKTree，选择最小查询距离的结点直到找到叶子结点。将叶子结点的向量作为后续在图中搜索的“种子”向量。

* Best-First Search
  * 把v的近邻点放入优先级队列Q
  * 从队列Q中取出第一个点v
  * 重复以上步骤，直到满足搜索条件

## Graph部分
Graph部分，通过先构建KNN图，再根据RNG Rule移除不符合要求的边，得到RNG。

### KNN图的构建
由于样本数据规模非常大，我们采用了一定的算法构建近似的KNN图，具体算法如下：
1. 随机划分一组子空间
2. 对该子空间内的点，利用Brute-Force方式，构建KNN子图。
3. 重复以上步骤N次。N越大，得到的KNN图越接近真实的KNN图。

![](./resource/partition.png)

每次随机划分一组子空间，会包含部分新的近邻点，而与之前划分的空间重叠的近邻点，可以将两组子空间构建的KNN子图连接成更大的KNN图。因此，划分次数越多，KNN子图越大，直到得到真实的KNN图为止。


* 算法来源：[Scalable k-NN graph construction for visual descriptors. Jing Wang, Jingdong Wang, Gang Zeng, Zhuowen Tu, Rui Gan, Shipeng Li. CVPR 2012.
](http://pages.ucsd.edu/~ztu/publication/cvpr12_knnG.pdf)


### KNG的构建
基于KNN图，我们需要根据RNG Rule删除不符合要求的边。这样做的目的是避免陷入局部最优。

RNG Rule：删除三角形中的最长的边。

对于KNN图，若点a, b, c相互连接，我们要分别计算3点的距离，删除最长的边。例如，图中需删除qb边，因为我们可以通过a从q访问到b。

![](./resource/RNGRule.png)



# SPTAG的使用