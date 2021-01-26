# Lab 4 - AllReduce的实现和优化

## 实验目的

1.	理解并行训练的原理和实现
2.	定制一个新的并行训练的通信压缩算法

## 实验环境

* Ubuntu 18.04
* PyTorch==1.5.0 (务必安装CPU版本)
* OpenMPI
* Horovod==0.19.4

## 实验原理

深度学习中，分布式训练算法和分布式训练系统的基本知识

## 实验内容

### 实验流程图

![](/imgs/Lab4-flow.png "Lab4 flow chat")

### 具体步骤

1.	安装依赖支持：OpenMPI, Horovod

2.	编写程序，使用Horovod库，增加数据并行训练支持

    1. 参照Horovod with PyTorch参考文档，修改 `mnist_basic.py` 文件, 另存为 `pytorch_mnist_horovod.py`，使用Horovod库实现数据并行
        - Mnist_basic.py原始文件地址：https://github.com/pytorch/examples/blob/master/mnist/main.py
        - Horovod with PyTorch文档地址：https://github.com/horovod/horovod/blob/master/docs/pytorch.rst
    2. 记录每个step的运行时间和正确率（accuracy）

3.	理解Horovod的执行逻辑，利用Numpy实现float8(8bit), float16(16bit)编码方案的压缩/解压缩

    1. 克隆GitHub上Horovod库
    2. 修改 `/horovod/torch/compression.py` 文件，增加Bit8Compressor和Bit16Compressor类，实现compress和decompress函数。（提示：torch.Tensor没有8-bit float类型支持，所以Bit8Compressor还需实现float32和float8类型的相互转化）

4.	修改Horovod库中代码，增加对float8(8bit), float16(16bit)格式的压缩

    1. 修改 `/horovod/torch/mpi_ops.py` 文件，利用Horovod内嵌的AllGather通信和压缩接口，增加对float8(8bit), float16(16bit)格式的压缩代码的调用。
    2. 重新build Horovod库。

5.	修改MNIST样例代码，增加压缩功能。

6.	测试代码正确性，比较原始代码、数据并行、加入压缩算法三者的性能差别。

7.	[选做项目] 利用C++/CUDA API实现更为高效的压缩/解压缩编码

## 实验报告

### 实验环境

||||
|--------|--------------|--------------------------|
|硬件环境|服务器数目|&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
||网卡型号、数目||
||GPU型号、数目||
||GPU连接方式||
|软件环境|OS版本||
||GPU driver、(opt. NIC driver)||
||深度学习框架<br>python包名称及版本||
||CUDA版本||
||||

### 实验结果

比较原始串行训练，用Horovod并行训练，加入压缩算法三者，在同样epoch条件下的训练时间和结果正确率。

Epoch size: ___________

|||||
|-----|-----|-----|-----|
| 训练算法 || &nbsp; &nbsp; &nbsp; &nbsp; 训练时间 &nbsp; &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; 结果正确率 &nbsp; &nbsp; &nbsp; &nbsp; |
|串行训练||||
| 用Horovod并行 | Device# == 2 |||
||Device# == 4|||
| float8(8bit)压缩 | Device# == 2 |||
|| Device# == 4 |||
| float16(16bit)压缩 | Device# == 2 |||
|| Device# == 4 |||
|||||

## 参考代码

### 安装Horovod

安装OpenMPI：`sudo apt install openmpi-bin`

安装Horovod：`python3 -m pip install horovod==0.19.4 --user`

### 利用Horovod并行化pytorch MNIST模型训练
1.	Device# == 1

    运行命令：`python3 pytorch_mnist_horovod.py`

2.	Device# == N  (e.g., N == 2, 4, 6, 8)

    运行命令：`horovodrun -n 2 python3 pytorch_mnist_horovod.py –hvd True `

    参考代码： https://github.com/horovod/horovod/blob/master/examples/pytorch_mnist.py

### 基于Horovod(v0.19.4)库增加bit-16和bit-8的并行训练的通信压缩算法

1.	Build Horovod

    运行命令：`HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITHOUT_GLOO=1 HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 python setup.py build`

2.	在horovod库中需要修改的文件和代码片段: bit8,bit16.git_diff

3.	执行压缩算法进行训练
 
    ```
    mpirun -n 2 python pytorch_mnist_compress.py --bit8-allreduce
    mpirun -n 2 python pytorch_mnist_compress.py --bit16-allreduce
    ```


## 参考资料

* Horovod with PyTorch 文档: https://github.com/horovod/horovod/blob/master/docs/pytorch.rst

* Horovod MNIST并行训练参考代码：https://github.com/horovod/horovod/blob/master/examples/pytorch_mnist.py
