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

 可以看到，a*b和np.multiply(a,b)的作用是一样的，都是点乘，即两个矩阵中相对应位置的数值相乘，element wise operation。
 
 而np.dot是一般的矩阵运算，要求a的列数和b的行数一样才能相乘，所以我们把b转置了一下，b本身是2行3列，b.T就是3行2列，a是2行3列，结果是2行2列。所以，一定不要被np.dot这个函数名字迷惑了，它不是点乘的意思。

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
