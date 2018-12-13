# AI基本原理简明教程的目录
Copyright © Microsoft Corporation. All rights reserved.
  适用于[License](https://github.com/Microsoft/ai-edu/blob/master/LICENSE.md)版权许可
  
## 写在前面，为什么要出这个系列的教程呢？

  总的说来，我们现在有了很多非常厉害的深度学习框架，比如tensorflow，pytorch，paddlepaddle，caffe2等等等等。然而，我们用这些框架在搭建我们自己的深度学习模型的时候，到底做了一些什么样的操作呢？我们试图去阅读框架的源码来理解框架到底帮助我们做了些什么，但是……很难！很难！很难！因为深度学习是需要加速啦，分布式计算啦，所以框架做了很多很多的优化，也让像我们这样的小白难以理解这些框架的源码。所以，为了帮助大家更进一步的了解神经网络模型的具体内容，我们整理了这样一个系列的教程。

对于这份教程的内容，如果没有额外的说明，我们通常使用如下表格的命名约定

| 符号 | 含义|
|:------------:|-------------|
| X | 输入样本 |
| Y | 输入样本的标签 |
| Z | 各层运算的结果|
| A | 激活函数结果|
| 大写字母 | 矩阵或矢量，如A,W,B|
| 小写字母 | 变量，标量，如a,w,b|

## 适用范围
  
  没有各种基础想学习却无从下手哀声叹气的玩家，请按时跟踪最新博客，推导数学公式，跑通代码，并及时提出问题，以求最高疗效；

  深度学习小白，有直观的人工智能的认识，强烈的学习欲望和需求，请在博客的基础上配合代码食用，效果更佳；

  调参师，训练过模型，调过参数，想了解框架内各层运算过程，给玄学的调参之路添加一点心理保障；

  超级高手，提出您宝贵的意见，给广大初学者指出一条明路！

## 前期准备

  环境：
  
  windows（Linux也行），python（最好用3），anaconda（或者自己装numpy之类的），tensorflow（嫌麻烦地请看这里[《AI应用开发实战 - 从零开始配置环境》](https://www.cnblogs.com/ms-uap/p/9123033.html)，tools for AI（按照链接教程走的就不用管这个了）。
  
  自己：

  清醒的头脑（困了的同学请自觉泡茶），纸和笔（如果像跟着推公式的话），闹钟（防止久坐按时起来转转），厚厚的衣服（有暖气的同学请忽略）

## 目录
+ [神经网络的基本工作原理](https://github.com/Microsoft/ai-edu/tree/master/B-教学案例与实践/B6-人工智能基本原理简明教程/1-神经网络的基本工作原理.md)
+ [神经网络中反向传播与梯度下降的基本概念](https://github.com/Microsoft/ai-edu/tree/master/B-教学案例与实践/B6-人工智能基本原理简明教程/2-反向传播与梯度下降.md)
+ [损失函数](https://github.com/Microsoft/ai-edu/tree/master/B-教学案例与实践/B6-人工智能基本原理简明教程/3-损失函数.md)
+ [单入单出的一层神经网络](https://github.com/Microsoft/ai-edu/tree/master/B-教学案例与实践/B6-人工智能基本原理简明教程/4-单入单出的一层神经网络.md)
+ [多入单出的一层神经网络](https://github.com/Microsoft/ai-edu/tree/master/B-教学案例与实践/B6-人工智能基本原理简明教程/5-多入单出的一层神经网络.md)
+ [多入多出的一层神经网络](https://github.com/Microsoft/ai-edu/tree/master/B-教学案例与实践/B6-人工智能基本原理简明教程/6-多入多出的一层神经网络.md)
+ [扩展阅读](https://github.com/Microsoft/ai-edu/tree/master/B-教学案例与实践/B6-人工智能基本原理简明教程/6-扩展阅读.md)
+ [分类函数](https://github.com/Microsoft/ai-edu/tree/master/B-教学案例与实践/B6-人工智能基本原理简明教程/7.1-分类函数.md)
+ [激活函数](https://github.com/Microsoft/ai-edu/tree/master/B-教学案例与实践/B6-人工智能基本原理简明教程/7.2-激活函数.md)
+ [单入单出的两层神经网络](https://github.com/Microsoft/ai-edu/tree/master/B-教学案例与实践/B6-人工智能基本原理简明教程/8-单入单出的两层神经网络.md)
+ [多入多出的两层神经网络](https://github.com/Microsoft/ai-edu/blob/master/B-%E6%95%99%E5%AD%A6%E6%A1%88%E4%BE%8B%E4%B8%8E%E5%AE%9E%E8%B7%B5/B6-%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86%E7%AE%80%E6%98%8E%E6%95%99%E7%A8%8B/9-%E5%A4%9A%E5%85%A5%E5%A4%9A%E5%87%BA%E7%9A%84%E4%B8%A4%E5%B1%82%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C.md)
+ 附录：[基本数学导数公式](https://github.com/Microsoft/ai-edu/tree/master/B-教学案例与实践/B6-人工智能基本原理简明教程/0-基本数学导数公式.md)