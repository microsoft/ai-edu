# Lab 5 - 配置Container进行云上训练或推理

## 实验目的

1. 理解Container机制
2. 使用Container进行自定义深度学习训练或推理

## 实验环境

* PyTorch==1.5.0
* Docker Engine

## 实验原理

计算集群调度管理，与云上训练和推理的基本知识

## 实验内容

### 实验流程图

![](/imgs/Lab5-flow.png "Lab5 flow chat")

### 具体步骤

1.	安装最新版Docker Engine，完成实验环境设置

2.	运行一个alpine容器

    1. Pull alpine docker image
    2. 运行docker container，并列出当前目录内容
    3. 使用交互式方式启动docker container，并查看当前目录内容
    4. 退出容器

3.	Docker部署PyTorch训练程序，并完成模型训练

    1. 编写Dockerfile：使用含有cuda10.1的基础镜像，编写能够运行MNIST样例的Dockerfile
    2. Build镜像
    3. 使用该镜像启动容器，并完成训练过程
    4. 获取训练结果

4.	Docker部署PyTorch推理程序，并完成一个推理服务

    1. 克隆TorchServe源码
    2. 编写基于GPU的TorchServe镜像
    3. 使用TorchServe镜像启动一个容器
    4. 使用TorchServe进行模型推理
    5. 返回推理结果，验证正确性


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

1.	使用Docker部署PyTorch MNIST 训练程序，以交互的方式在容器中运行训练程序。提交以下内容：

    1. 创建模型训练镜像，并提交Dockerfile
    2. 提交镜像构建成功的日志
    3. 启动训练程序，提交训练成功日志（例如：MNIST训练日志截图）

<br/>

<br/>

<br/>

<br/>


2.	使用Docker部署MNIST模型的推理服务，并进行推理。提交以下内容：
    1. 创建模型推理镜像，并提交Dockerfile
    2. 启动容器，访问TorchServe API，提交返回结果日志
    3. 使用训练好的模型，启动TorchServe，在新的终端中，使用一张手写字体图片进行推理服务。提交手写字体图片，和推理程序返回结果截图。

<br/>

<br/>

<br/>

<br/>

<br/>

## 参考代码

本次实验基本教程:

* [1. 实验环境设置](./setup.md)
* [2. 运行你的第一个容器 - 内容，步骤，作业](./alpine.md)
* [3. Docker部署PyTorch训练程序 - 内容，步骤，作业](./train.md)
* [4. Docker部署PyTorch推理程序 - 内容，步骤，作业](./inference.md)
* [5. 进阶学习](./extend.md)

## 参考资料

* [Docker Tutorials and Labs](https://github.com/docker/labs/)
* [A comprehensive tutorial on getting started with Docker!](https://github.com/prakhar1989/docker-curriculum)
* [Please-Contain-Yourself](https://github.com/dylanlrrb/Please-Contain-Yourself)
* [Create TorchServe docker image](https://github.com/pytorch/serve/tree/master/docker)


