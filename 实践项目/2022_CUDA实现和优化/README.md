# 2022年微软开源学习社群实践项目——CUDA实现和优化
## 实验目的：
•	理解DNN框架中的张量运算在GPU加速器上的加速原理
•	通过CUDA实现和优化一个定制化张量运算

开源链接：https://github.com/microsoft/AI-System/tree/main/Labs/BasicLabs/Lab3

## 技能要求：
•	学习git/GitHub基本操作
•	使用markdown编写文档

## 实验环境：
•	PyTorch==1.5.0
•	CUDA 10.0

## 任务详情及预期成果
### 实验流程图
![lab3-flow.png](https://note.youdao.com/yws/res/90/WEBRESOURCEad9964ce391af14498cacc1dac70bbce)
### 具体步骤
1.	理解PyTorch中Linear张量运算的计算过程，推导计算公式
2.	了解GPU端加速的原理，CUDA内核编程和实现一个kernel的原理
3.	实现CUDA版本的定制化张量运算
I.	编写.cu文件，实现矩阵相乘的kernel
II.	在上述.cu文件中，编写使用cuda进行前向计算和反向传播的函数
III.	基于C++ API，编写.cpp文件，调用上述函数，实现Linear张量运算和前向计算和反向传播
IV.	将代码生成Python的C++扩展
V.	使用基于C++的函数扩展，实现自定义Linear类模块的前向计算和反向传播函数
VI.	运行程序，验证网络正确性
4.	使用profile比较网络性能：基于C++ API，比较有无CUDA对张量运算性能的影响
5.	[选做，加分]实现基于CUDA的卷积层（Convolutional）自定义张量运算

## 项目提交方式
提交项目至GitHub“微软人工智能教育与学习共建社区“专用issue，需要有文档和代码，文档中给出代码的具体运行步骤

## 项目时间安排
项目4月中旬发布，6月中旬验收。项目时长约为2个月