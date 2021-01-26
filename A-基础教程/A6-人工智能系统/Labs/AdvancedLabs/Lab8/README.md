# Lab 8 - 自动机器学习系统练习

## 实验目的

通过试用 NNI 了解自动机器学习，熟悉自动机器学习中的基本概念

## 实验环境

* Ubuntu
* Python==3.7.6
* NNI==1.8
* PyTorch==1.5.0

## 实验原理

在本实验中，我们将处理 CIFAR-10 图片分类数据集。基于一个表现较差的基准模型和训练方法，我们将使用自动机器学习的方法进行模型选择和优化、超参数调优，从而得到一个准确率较高的模型。

## 实验内容

### 实验流程图

![](/imgs/Lab8-flow.png "Lab8 flow chat")

### 具体步骤

1. 熟悉 PyTorch 和 CIFAR-10 图像分类数据集。可以先阅读教程：https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html 
   
2. 熟悉 NNI 的基本使用。阅读教程：https://nni.readthedocs.io/en/latest/Tutorial/QuickStart.html 
   
3. 运行CIFAR-10代码并观察训练结果。在实验目录下，找到 `hpo/main.py`，运行程序，记录模型预测的准确率。
   
4. 手动参数调优。通过修改命令行参数来手动调整超参，以提升模型预测准确率。记录调整后的超参名称和数值，记录最终准确率。
   
   **注：**
   main.py 暴露大量的命令行选项，可以进行调整，命令行选项可以直接从代码中查找，或通过 `python main.py -h` 查看。例如，`--model`（默认是 resnet18），`--initial_lr`（默认是 0.1），`--epochs`（默认是 300）等等。一种简单的方法是通过手工的方法调整参数（例如 `python main.py --model resnet50 --initial_lr 0.01`）然后根据结果再做调整。
5. 使用 NNI 加速参数调优过程。
   
    1. 参考NNI的基本使用教程，安装NNI（建议在Linux系统中安装NNI并运行实验）。
    2. 参照NNI教程运行 `mnist-pytorch` 样例程序（程序地址： https://github.com/microsoft/nni/tree/master/examples/trials/mnist-pytorch  ），测试安装正确性，并熟悉NNI的基本使用方法。
    3. 使用NNI自动调参功能调试hpo目录下CIFAR-10程序的超参。创建 `search_space.json` 文件并编写搜索空间（即每个参数的范围是什么），创建 `config.yml` 文件配置实验（可以视资源量决定搜索空间的大小和并行量），运行程序。在 NNI 的 WebUI 查看超参搜索结果，记录结果截图，并记录得出最好准确率的超参配置。
   
6.	（可选）上一步中进行的模型选择，是在若干个前人发现的比较好的模型中选择一个。此外，还可以用自动机器学习的方法选择模型，即网络架构搜索（NAS）。请参考nas目录下 `model.py`，采用 DARTS 的搜索空间，选择合适的 Trainer，进行搜索训练。记录搜索结果架构，并用此模型重新训练，记录最终训练准确率。

**注：** 搜索完成后得到的准确率并不是实际准确率，需要使用搜索到的模型重新进行单独的训练。具体请参考 NNI NAS 文档：https://nni.readthedocs.io/en/latest/nas.html


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

1.	记录不同调参方式下，cifar10程序训练结果的准确率。

||||
|---------|-----------------|------------|
| 调参方式 | &nbsp; &nbsp; 超参名称和设置值 &nbsp; &nbsp; | &nbsp; &nbsp; 模型准确率 &nbsp; &nbsp; |
| &nbsp; <br /> &nbsp; 原始代码 &nbsp; <br /> &nbsp; |||
| &nbsp; <br /> &nbsp; 手动调参 &nbsp; <br /> &nbsp; |||
| &nbsp; <br /> &nbsp; NNI自动调参 &nbsp; <br /> &nbsp; |||
| &nbsp; <br /> &nbsp; 网络架构搜索 <br />&nbsp; &nbsp; （可选） <br /> &nbsp; |||
||||
2.	提交使用NNI自动调参方式，对 main.py、search_space.json、config.yml 改动的代码文件或截图。

<br />

<br />

<br />

<br />

<br />

3.	提交使用NNI自动调参方式，Web UI上的结果截图。

<br />

<br />

<br />

<br />

<br />

4.	（可选）提交 NAS 的搜索空间、搜索方法和搜索结果（得到的架构和最终准确率）。

<br />

<br />

<br />

<br />

<br />


## 参考代码

### 自动调参

代码位置：`Lab8/hpo`

参考答案：`Lab8/hpo-answer`

### 网络架构搜索（NAS）

代码位置：`Lab8/nas`


## 参考资料

* Cifar10简介：https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html 
* NNI文档：https://nni.readthedocs.io/en/latest/ 
* NNI mnist-pytorch代码：https://github.com/microsoft/nni/tree/v1.9/examples/trials/mnist-pytorch
* NNI NAS 文档：https://nni.readthedocs.io/en/latest/nas.html 
* DARTS GitHub：https://github.com/quark0/darts 
