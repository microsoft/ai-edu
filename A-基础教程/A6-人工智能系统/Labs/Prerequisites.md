# Setup Environment

## 实验环境要求

### 操作系统

`Ubuntu 18.04 LTS x86_64`

**注：** 感兴趣的windows用户可以尝试使用 `windows 10 x64 with Ubuntu18.04 LTS sub-system（Windows 10 Insider Preview build 18975 (Slow) or later for WSL 2)`, 如有任何问题和建议，可以通过 `issues` 反馈给我们。

本实验中测试均在 Ubuntu 18.04 中完成。

### 编程语言

`python3.7.6` (`Anaconda3`环境)

### 学习框架

`PyTorch==1.5.0`

### 硬件环境（由低到高排序）

1.	单机 
2.	单机，单GPU（with CUDA 10.1）
3.	单机，多GPU（with CUDA 10.1）
4.	多机，多GPU（with CUDA 10.1）

<br/>


## 实验环境搭建

### 安装anaconda3

安装教程：https://docs.anaconda.com/anaconda/install/linux/#installation

下载地址：https://www.anaconda.com/distribution/#linux

1.	下载命令：(注意：$ 为linux系统下的命令提示符，不属于命令部分)
    ```
    $ wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
    $ bash Anaconda3-2020.02-Linux-x86_64.sh
    ```
    **注：** 请按照命令行提示安装anaconda，所有选项输入yes)

2.	激活conda环境：
    ```
    $ source ~/.bashrc
    ```

3.	测试是否安装成功：
    ```
    $ conda -V
    ```
    若安装成功，则会显示：  conda 4.8.2

### 安装python3.7.6（若使用Anaconda3的base环境，默认是python3.7.6，则不需额外安装）

1.	创建新的conda环境：
    ```
    $ conda create -n py37 python=3.7.6
    ```
2.	激活python3.7:
    ```
    $ conda activate py37
    ```

### 安装gcc（如果机器已经安装gcc，请忽略）
```
$ sudo apt-get update
$ sudo apt-get install build-essential
```

### 安装pytorch (version 1.5.0)

CPU版本： 
```
$ conda install pytorch==1.5.0 torchvision==0.6.0 cpuonly -c pytorch
```

GPU（CUDA10.1）版本：
```
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
```
 
