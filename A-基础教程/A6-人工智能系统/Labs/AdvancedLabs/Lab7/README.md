# Lab 7 - 分布式训练任务练习

## 实验目的

1.	学习使用Horovod库。
2.	通过调用不同的通信后端实现数据并行的并行/分布式训练，了解各种后端的基本原理和适用范围。
3.	通过实际操作，灵活掌握安装部署。

## 实验环境

* Ubuntu 18.04
* CUDA 10.0
* PyTorch==1.5.0
* Horovod==0.19.4

## 实验原理

通过测试MPI、NCCL、Gloo、oneCCL后端完成相同的allreduce通信，通过不同的链路实现数据传输。

## 实验内容

### 实验流程图

![](/imgs/Lab7-flow.png "Lab7 flow chat")

### 具体步骤

1.	安装依赖支持：OpenMPI, Horovod。

2.	运行Horovod MNIST测试用例(`Lab7/pytorch_mnist_horovod.py`)，验证Horovod正确安装。

3.	按照MPI/Gloo/NCCL的顺序，选用不同的通信后端，测试不同GPU数、不同机器数时，MNIST样例下iteration耗时和吞吐率，记录GPU和机器数目，以及测试结果，并完成表格绘制。

    1. 安装MPI，并测试多卡、多机并行训练耗时和吞吐率。可参考如下命令：
        ```
        //单机多CPU
        $horovodrun -np 2 python pytorch_mnist_horovod.py --no-cuda

        //多机单GPU
        $horovodrun -np 4 -H server1:1,server2:1,server3:1,server4:1 python pytorch_mnist_horovod_basic.py
        ```

    2. 测试Gloo下的多卡、多机并行训练耗时。

    3. 安装NCCL2后重新安装horovod并测试多卡、多机并行训练耗时和吞吐率。


4.	（可选）安装支持GPU通信的MPI后重新安装horovod并测试多卡、多机并行训练耗时和吞吐率。

    ```
    $ HOROVOD_GPU_ALLREDUCE=MPI pip install --no-cache-dir horovod
    ```

5.	（可选）若机器有Tesla/Quadro GPU + RDMA环境，尝试设置GPUDirect RDMA 以达到更高的通信性能

6.	统计数据，绘制系统的scalability曲线H

7.	（可选）选取任意RNN网络进行并行训练，测试horovod并行训练耗时和吞吐率。


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
||NCCL版本||
||||

### 实验结果
1.	测试服务器内多显卡加速比

    |||||||
    |-----|-----|-----|-----|------|------|
    | 通信后端 | 服务器数量 | 每台服务器显卡数量 | 平均每步耗时 | 平均吞吐率 | 加速比 |
    | MPI | 1 | | | | |
    | MPI | 1 | | | | |
    | ... | | | | | |
    | Gloo | 1 | | | | |
    | Gloo | 1 | | | | |
    | ... | | | | | |
    | NCCL | 1 | | | | |
    | NCCL |1  | | | | |
    | ... | | | | | |
    |||||||

2.	测试服务器间加速比

    |||||||
    |-----|-----|-----|-----|------|------|
    | 通信后端 | 服务器数量 | 每台服务器显卡数量 | 平均每步耗时 | 平均吞吐率 | 加速比 |
    | MPI |  | 1 ||||
    | MPI |  | 1 ||||
    | Gloo |  | 1 ||||
    | Gloo |  | 1 ||||
    | NCCL |  | 1 ||||
    | NCCL |  | 1 ||||
    |||||||


3.	总结加速比的图表、比较不同通信后端的性能差异、分析可能的原因

<br />

<br />

<br />

<br />

4.	（可选）比较不同模型的并行加速差异、分析可能的原因（提示：计算/通信比）

<br />

<br />

<br />

<br />

## 参考代码

1. 安装依赖支持

    安装OpenMPI：`sudo apt install openmpi-bin`

    安装Horovod：`python3 -m pip install horovod==0.19.4 --user`

2. 验证Horovod正确安装

    运行mnist样例程序
    ```
    python pytorch_mnist_horovod_basic.py
    ```
3. 选用不同的通信后端测试命令

    1.	安装MPI，并测试多卡、多机并行训练耗时和吞吐率。
        ```
        //单机多CPU
        $horovodrun -np 2 python pytorch_mnist_horovod.py --no-cuda

        //单机多GPU
        $horovodrun -np 2 python pytorch_mnist_horovod.py

        //多机单GPU
        $horovodrun -np 4 -H server1:1,server2:1,server3:1,server4:1 python pytorch_mnist_horovod_basic.py

        //多机多CPU
        $horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python pytorch_mnist_horovod_basic.py --no-cuda

        //多机多GPU
        $horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python pytorch_mnist_horovod_basic.py

        ```
    2.	测试Gloo下的多卡、多机并行训练耗时。
        ```
        $horovodrun --gloo -np 2 python pytorch_mnist_horovod.py --no-cuda

        $horovodrun -np 4 -H server1:1,server2:1,server3:1,server4:1 python pytorch_mnist_horovod_basic.py

        $horovodrun –gloo -np 16 -H server1:4,server2:4,server3:4,server4:4 python pytorch_mnist_horovod_basic.py --no-cuda

        ```
    3.	安装NCCL2后重新安装horovod并测试多卡、多机并行训练耗时和吞吐率。
        ```
        $HOROVOD_GPU_OPERATIONS=NCCL pip install --no-cache-dir horovod

        $horovodrun -np 2 -H server1:1,server2:1 python pytorch_mnist_horovod.py
        ```
    4.	安装支持GPU通信的MPI后重新安装horovod并测试多卡、多机并行训练耗时和吞吐率。
        ```
        HOROVOD_GPU_ALLREDUCE=MPI pip install --no-cache-dir horovod
        ```



## 参考资料

* PyTorch MNIST测试用例：https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py 

* Horovod on GPU: https://github.com/horovod/horovod/blob/master/docs/gpus.rst 

* NCCL2 download: https://developer.nvidia.com/nccl/nccl-download 

* OpenMPI: https://www.open-mpi.org/software/ompi/v4.0/ 
