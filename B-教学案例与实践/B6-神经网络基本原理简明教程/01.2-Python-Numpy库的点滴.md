# Python中的Numpy的基本知识

Copyright © Microsoft Corporation. All rights reserved.
  适用于[License](https://github.com/Microsoft/ai-edu/blob/master/LICENSE.md)版权许可

以下列出一些关于Numpy矩阵运算的基本知识和坑点。

首先需要在命令行中安装Numpy库和绘图库（可选）：

```
pip install numpy
pip install matplotlib
```

然后在python文件的第一行，加入对它们的引用：

```Python
import numpy as np
import matplotlib.pyplot as plt
```

- 基本矩阵运算

```Python
a=np.array([1,2,3,4,5,6]).reshape(2,3)
b=np.array([1,1,1,1,1,1]).reshape(2,3)

print("a:")
print(a)

print("b:")
print(b)

print("a*b:")
print(a*b)

print("np.multiply(a,b):")
print(np.multiply(a,b))

print("np.dot(a,b.T)")
print(np.dot(a,b.T))

# 以下这个命令会出错
print(np.dot(a,b))
```

结果：
```
a:
[[1 2 3]
 [4 5 6]]
b:
[[1 1 1]
 [1 1 1]]
a*b:
[[1 2 3]
 [4 5 6]]
np.multiply(a,b):
[[1 2 3]
 [4 5 6]]
np.dot(a,b.T)
[[ 6  6]
 [15 15]]
 ```

 可以看到，a*b和np.multiply(a,b)的作用是一样的，都是点乘，即两个矩阵中相对应位置的数值相乘，element wise operation。它的输出与相乘矩阵的尺寸一致。
 
 而np.dot是标准的矩阵运算，。如果输入是(3x2)x(2x4)，则输出为3x4。要求a的列数和b的行数一样才能相乘，所以我们把b转置了一下，b本身是2行3列，b.T就是3行2列，a是2行3列，结果是2行2列。所以，一定不要被np.dot这个函数名字迷惑了，它不是点乘的意思。

```Python
a=np.array([1,2,3])
b=np.array([1,1,1]).reshape(1,3)
print(a.shape)
print(a*b)
a=a.reshape(3,1)
print(a.shape)
print(a*b)
```

结果：
```
(3,)
[[1 2 3]]
(3, 1)
[[1 1 1]
 [2 2 2]
 [3 3 3]]
```
第一次定义a时，是一个1维列向量，shape=(3,)，用a*b得到的结果是shape=(1,3)的矩阵\[\[1 2 3]]。
后来把a.reshape(3,1)3行1列的二维矩阵，虽然表面看起来形式没变，但是在与b点乘后，得到了一个(3,3)的矩阵。
为了避免一些错误，最好在每次矩阵运算前，都把两个矩阵reshape成一个二维矩阵（或多维矩阵）。

# 神经网络中的计算过程

- w=(3x2)
w=np.array([1,2,3,4,5,6]).reshape(3,2)
```
[[1 2]
 [3 4]
 [5 6]]
 ```
- b=(3x1)
b=np.array([1,2,3]).reshape(3,1)
```
[[1]
 [2]
 [3]]
```

- x=(2x4)(2个特征值,4个样本)
x=np.array([2,3,4,5,6,7,8,9]).reshape(2,4)
```
[[2 3 4 5]
 [6 7 8 9]]
```
- c=np.dot(w,x)
```
[[14 17 20 23]
 [30 37 44 51]
 [46 57 68 79]]
```
- z=c+b = np.dot(w,x) + b
注意：这里加法有对b的列广播（自动扩充b为4x3，通过复制b的值为3列）
```
b=
[[1 1 1 1]
 [2 2 2 2]
 [3 3 3 3]]

z=
[[15 18 21 24]
 [32 39 46 53]
 [49 60 71 82]]
```
- y=np.array([6,5,4,3]).reshape(1,4)
4个样本的标签值
```
[[6 5 4 3]]
```
- dz = z - y
注意：这里减法有对z的广播(通过复制y的值为3行)
```
z=
[[15 18 21 24]
 [32 39 46 53]
 [49 60 71 82]]

y=
[[6 5 4 3]
 [6 5 4 3]
 [6 5 4 3]]

dz=
[[ 9 13 17 21]
 [26 34 42 50]
 [43 55 67 79]]
```
- db = dz.sum(axis=1,keepdims=True)/4
4是样本数。axis=1, 按列相加，即一行内的所有列元素相加。除以4是广播。
```
[[15.]
 [38.]
 [61.]]
```
- dw=np.dot(dz,x.T)/4
x.T是x的转置。除以4是广播。 dz=3x4, x.T=4x2, 结果是3x2，正好是w的shape。
```
[[ 57.5 117.5]
 [143.  295. ]
 [228.5 472.5]]
```
- w = w - 0.1*dw
```
[[ -4.75  -9.75]
 [-11.3  -25.5 ]
 [-17.85 -41.25]]
 ```
 - b = b - 0.1*db
```
[[-0.5]
 [-1.8]
 [-3.1]]
 ```
