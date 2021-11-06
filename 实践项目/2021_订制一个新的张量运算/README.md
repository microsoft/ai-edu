# 2021年微软开源学习社群实践项目：定制一个新的张量运算
## 项目简介：
本实践案例主要任务为定制一个新的张量运算。通过完成本案例，参与者可以了解深度学习框架执行机制，理解DNN框架中的张量算子的原理，并在基于不同方法实现新的张量运算的同时比较其性能差异。

## 实验环境：
PyTorch==1.5.0

## 实验原理：
- 深度神经网络中的张量运算原理
- PyTorch中基于Function和Module构造张量的方法
- 通过C++扩展编写Python函数模块

## 具体步骤：
- 在MNIST的模型样例中，选择线性层（Linear）张量运算进行定制化实现。
- 理解PyTorch构造张量运算的基本单位：Function和Module。
- 基于Function和Module的Python API重新实现Linear张量运算。
> - 修改MNIST样例代码
> - 基于PyTorch Module编写自定义的Linear 类模块
> - 基于PyTorch Function实现前向计算和反向传播函数
> - 使用自定义Linear替换网络中nn.Linear() 类
> - 运行程序，验证网络正确性
- 理解PyTorch张量运算在后端执行原理
- 实现C++版本的定制化张量运算
- 基于C++，实现自定义Linear层前向计算和反向传播函数，并绑定为Python模型
> - 将代码生成python的C++扩展
> - 使用基于C++的函数扩展，实现自定义Linear类模块的前向计算和反向传播函数
> - 运行程序，验证网络正确性
- 使用profiler比较网络性能：比较原有张量运算和两种自定义张量运算的性能
- 【可选实验，加分】实现卷积层（Convolutional）的自定义张量运算

附：案例实现流程图
![](./zhangliang.png)
 
## 项目提交方式：
提交项目至GitHub“微软人工智能教育与学习共建社区”专用[Issue](https://github.com/microsoft/ai-edu/issues/680)，需要有文档和代码，文档中给出代码的具体运行步骤。

## 案例参考地址：
AI-System/Labs/BasicLabs/Lab2 at main · microsoft/AI-System · GitHub
