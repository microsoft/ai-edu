# Lab 3 - CUDA实现和优化

## 实验目的

1.	理解DNN框架中的张量运算在GPU加速器上加速原理。
2.	通过CUDA实现和优化一个定制化张量运算。

## 实验环境

* PyTorch==1.5.0

* CUDA 10.0

## 实验原理

1.	矩阵运算与计算机体系结构
2.	GPU加速器的加速原理

## 实验内容

### 实验流程图

![](/imgs/Lab3-flow.png "Lab3 flow chat")

### 具体步骤

1.	理解PyTorch中Linear张量运算的计算过程，推导计算公式

2.	了解GPU端加速的原理，CUDA内核编程和实现一个kernel的原理

3.	实现CUDA版本的定制化张量运算

    1. 编写.cu文件，实现矩阵相乘的kernel
   
    2. 在上述.cu文件中，编写使用cuda进行前向计算和反向传播的函数
   
    3. 基于C++ API，编写.cpp文件，调用上述函数，实现Linear张量运算的前向计算和反向传播。

    4. 将代码生成python的C++扩展

    5. 使用基于C++的函数扩展，实现自定义Linear类模块的前向计算和反向传播函数

    6. 运行程序，验证网络正确性

4.	使用profiler比较网络性能：基于C++API，比较有无CUDA对张量运算性能的影响

5.	【可选实验，加分】实现基于CUDA的卷积层（Convolutional）自定义张量运算

## 实验报告

### 实验环境

||||
|--------|--------------|--------------------------|
|硬件环境|CPU（vCPU数目）|&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
||GPU(型号，数目)||
|软件环境|OS版本||
||深度学习框架<br>python包名称及版本||
||CUDA版本||
||||

### 实验结果

|||
|---------------|---------------------------|
| 实现方式（Linear层为例）| &nbsp; &nbsp; &nbsp; &nbsp; 性能评测 |
|<br/> <br/>CPU only<br/> <br/>&nbsp;|&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
|<br/> <br/>With CUDA<br/> <br/>&nbsp;||
||||

## 参考代码

代码位置：`Lab3/mnist_custom_linear_cuda.py`

运行命令：
```
cd mylinear_cuda_extension
python setup install --user
cd ..& python mnist_custom_linear_cuda.p
```

## 参考资料

* CUDA Programming model: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html 
* An Even Easier Introduction to CUDA: https://devblogs.nvidia.com/even-easier-introduction-cuda/ 
* CUSTOM C++ AND CUDA EXTENSIONS: https://pytorch.org/tutorials/advanced/cpp_extension.html
