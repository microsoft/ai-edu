# Lab 9 - 强化学习系统练习（RLlib的实践与应用）

## 实验目的

1.	通过快速上手RLlib
2.	理解分布式强化学习系统的各模块的构成
3.	理解强化学习的分布式算法及其性能

## 实验环境

* Linux集群（至少两台Linux机器）
* Python==3.7.6
* ray
* rllib
* PyTorch==1.5.0

## 实验原理

RLlib是由UC Berkeley发起的一个开源的强化学习（Reinforcement Learning，简称RL）框架， 提供了高度可扩展性的API， 可以让用户在其框架上实现不同的RL算法，或者将已有的算法跑在分布式平台上。Rllib既可以支持多种多样不同的RL算法（例如DQN, policy grident, SAC, DDPG等），也支持连接各种不同的环境（例如gym, MuJoCo等）， 同时也支持把不同的分布式RL算法（例如apex-dqn，IMPALA等）跑在集群上。RLlib支持pytroch和tensorflow/tensorflow eager等不同的深度学习框架。

![](/imgs/RLlib-architecture.png "RLlib architecture")

**注：** 上图出自https://docs.ray.io/en/latest/rllib.html

本实验通过不同的配置, 理解不同的分布式强化学习算法在不同并行条件下的不同环境的表现。

## 实验内容

### 实验流程图

![](/imgs/Lab9-flow.png "Lab9 flow chat")

### 具体步骤

1.	安装环境依赖包 `ray` 和 `rllib` ，并测试是否安装成功。
    ```
    pip install -U ray
    pip install ray[rllib] 
    ```

2.	配置分布式RLlib环境, 并检测分布式环境是否成功
    1. 参考如下命令，配置主节点（master节点）
        ```
        ray start --head --redis-port=6666
        ```
    注：

    a.	该port为ray预留的可以被其他机器访问的端口

    b.	可以通过ssh 访问机器，或直接登录到机器进行配置

    2. 参考如下命令，配置工作节点（worker节点）
        ```
        ray start --address=<master_address> 
        ```
        **注：** master_address指的是主节点的IP地址 

3.	配置不同的脚本，测试不同算法对应不同并行条件/不同环境下的收敛速度。至少挑选一种分布式算法，并测试其worker并行数目为4，8，16的情况下在至少两个Atari环境下的收敛情况，提交配置文件和对应的启动脚本文件。

    1. 在算法为apex-dqn，并行条件为worker数目为2，4，16的情况下，测试在pong的环境下的收敛情况。
   
    2. 在算法为apex-dppg，并行条件为worker数目为2，4，16的情况下，测试在pendulum的环境下的收敛情况。
   
    3. 在算法为impala，并行条件为worker数目为2，4，16的情况下，测试在cartpole的环境下的收敛情况。
 
4.	收敛结果的分析，包括不同并行条件/环境下的不同算法的收敛的time和reward。总结成表格，并画出对应的学习曲线。


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

1.	提交不同算法、环境和并行条件（worker数目）下，配置文件和启动脚本。

<br />

<br />

<br />

<br />

<br />

2.	收敛结果的分析 

    1. 提交不同config的运行输出文件

        <br />

        <br />

        <br />

        <br />

        <br />


    2. 填写不同的算法在不同并行条件/环境下，收敛所需要的time和reward表格

        ||||||
        |---|---|---|---|---|
        | 算法 | 环境 | 并行条件 | &nbsp; &nbsp; &nbsp; Time &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; Reward &nbsp; &nbsp; &nbsp; |
        | apex-dqn | pong | 2 |||
        ||| 4 |||
        ||| 16 |||
        | apex-dppg | pendulum | 2 |||
        ||| 4 |||
        ||| 16 |||
        | Imapla | cartpole | 2 |||
        ||| 4 |||
        ||| 16 |||
        ||||||

3. 根据b的表格生成不同的学习曲线 

<br />

<br />

<br />

<br />

<br />


## 参考代码

### 安装依赖包
```
pip install -U ray
pip install ray[rllib]
```

### 检测依赖包是否安装成功 
1.	测试ray
```
git clone https://github.com/ray-project/ray.git 
cd ray 
python -m pytest -v python/ray/tests/test_mini.py 
```

2.	测试rllib 
```
rllib train --run=PPO --env=CartPole-v0 
```

### 检测分布式的rllib的环境是否配置成功 

1.	配置主节点，ssh到主节点进行配置：
    ``` 
    ray start --head --redis-port=6666 
    ```
    该`port`为 ray 预留的可以被其他机器访问的端口 

2.	配置工作节点，登录到每一台其他节点上进行配置： 
    ```
    ray start --address=<master_address> 
    ```
    `master_address` 指的是主节点的IP地址 

### 参考的不同分布式算法对应不同环境/并行条件的配置

代码位置：`Lab9/config`

参考命令：
```
cd Lab9 
rllib train -f config/xxx-xxx.yaml
```


## 参考资料

* Ray GitHub仓库：https://github.com/ray-project/ray 
  
* Ray和RLlib的官方文档：https://docs.ray.io/en/latest/index.html 
  
* RLlib编写config参考链接： https://docs.ray.io/en/master/rllib-training.html 
