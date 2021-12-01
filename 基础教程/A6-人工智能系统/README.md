# 人工智能系统

[English](https://github.com/microsoft/AI-System/blob/main/README_en.md)

本课程的中文名称设定为 **人工智能系统**，主要讲解支持人工智能的计算机系统设计，对应的英文课程名称为 **System for AI**。本课程中将交替使用一下词汇：**人工智能系统**，**AI-System** 和 **System for AI**。

本课程为[微软人工智能教育与共建社区](https://github.com/microsoft/ai-edu)中规划的人工智能相关教程之一，在[A-基础教程](https://github.com/microsoft/ai-edu/tree/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B)模块下，课程编号和名称为 *A6-人工智能系统*。

欢迎访问[微软人工智能教育与共建社区](https://github.com/microsoft/ai-edu)的[A-基础教程](https://github.com/microsoft/ai-edu/tree/master/A-%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B)模块访问更多相关内容。

## 人工智能系统课程设立背景

近年来人工智能特别是深度学习技术得到了飞速发展，这背后离不开计算机硬件和软件系统的不断进步。在可见的未来，人工智能技术的发展仍将依赖于计算机系统和人工智能相结合的共同创新模式。需要注意的是，计算机系统现在正以更大的规模和更高的复杂性来赋能于人工智能，这背后不仅需要更多的系统上的创新，更需要系统性的思维和方法论。与此同时，人工智能也反过来为设计复杂系统提供支持。

我们注意到，现在的大部分人工智能相关的课程，特别是深度学习和机器学习相关课程主要集中在相关理论、算法或者应用，与系统相关的课程并不多见。我们希望人工智能系统这门课能让人工智能相关教育变得更加全面和深入，以共同促进人工智能与系统交叉人才的培养。


## 人工智能系统课程设立目的

本课程主要为本科生高年级和研究生设计，帮助学生：

1. 完整的了解支持深度学习的计算机系统架构，并通过实际的问题，来学习深度学习完整生命周期下的系统设计。
 
2. 介绍前沿的系统和人工智能相结合的研究工作，包括AI for Systems and Systems for AI，以帮助高年级的本科生和研究生更好的寻找和定义有意义的研究问题。

3. 从系统研究的角度出发设计实验课程。通过操作和应用主流和最新的框架、平台和工具来鼓励学生动手实现和优化系统模块，以提高解决实际问题的能力，而不仅仅是了解工具使用。

**先修课程:** C/C++/Python, 计算机体系结构，算法导论


## 人工智能系统课程的设计与特点

课程主要包括以下三大模块：

第一部分，是人工智能的基础知识和人工智能系统的全栈概述；以及深度学习系统的系统性设计和方法学。

第二部分，为高级课程，包括最前沿的系统和人工智能交叉的研究领域。

第三部分，是与之配套的实验课程，包括最主流的框架、平台和工具，以及一系列的实验项目。

第一部分的内容将集中在基础知识，而其他两部分的内容将随着学术界和工业界的技术进步而动态调整。后两部分的内容将以模块化的形式组织，以利于调整或与其他CS的课程（比如编译原理等）相结合，作为高级讲义或者实习项目。

本课程的设计也会借助微软亚洲研究院在人工智能和系统交叉领域的研究成果和经验，其中包括微软及研究院开发的一部分平台和工具。课程也鼓励其他学校和老师根据自己的需求添加和调整更多的高级课题，或者其他的实验。


## 人工智能系统课程大纲

### [课程部分](./Lectures)

*基础课程*
||||
|---|---|---|
|课程编号|讲义名称|备注|
|1|课程介绍|课程概述和系统/AI基础|
|2|人工智能系统概述|人工智能系统发展历史，神经网络基础，人工智能系统基础|
|3|深度神经网络计算框架基础|反向传播和自动求导，张量，有向无环图，执行图 <br>论文和系统：PyTorch, TensorFlow|
|4|矩阵运算与计算机体系结构|矩阵运算，CPU/SIMD, GPGPU, ASIC/TPU <br>论文和系统：Blas, TPU|
|5|分布式训练算法|数据并行，模型并行，分布式SGD <br>论文和系统：PipeDream|
|6|分布式训练系统|MPI, parameter servers, all-reduce, RDMA <br>论文和系统: Horovod|
|7|异构计算集群调度与资源管理系统|集群上运行DNN任务：容器，资源分配，调度 <br>论文和系统：Kubeflow, OpenPAI, Gandiva|
|8|深度学习推导系统|效率，延迟，吞吐量，部署 <br>论文和系统：TensorRT, TensorFlowLite, ONNX|
||||

*高阶课程*
||||
|---|---|---|
|课程编号|讲义名称|备注|
|9|计算图的编译与优化|IR，子图模式匹配，矩阵乘和内存优化 <br>论文和系统：XLA, MLIR, TVM, NNFusion|
|10|神经网络的压缩与稀疏化优化|模型压缩，稀疏化，剪枝|
|11|自动机器学习系统|超参调优，神经网络结构搜索（NAS）<br>论文和系统：Hyperband, SMAC, ENAX, AutoKeras, NNI|
|12|强化学习系统|RL理论，RL系统 <br>论文和系统：AC3, RLlib, AlphaZero|
|13|安全与隐私|联邦学习，安全，隐私 <br>论文和系统：DeepFake|
|14|利用人工智能来优化计算机系统问题|AI用于传统系统问题，AI用于系统算法 <br>论文和系统：Learned Indexes, Learned query path|
||||


### [实验部分](./Labs)

*基础实验*
||||
|---|---|---|
|实验编号|实验名称|备注|
|实验 1|框架及工具入门示例|
|实验 2|定制一个新的张量运算|
|实验 3|CUDA实现和优化|
|实验 4|AllReduce的实现或优化|
|实验 5|配置Container来进行云上训练或推理准备|
||||

*高阶实验*
||||
|---|---|---|
|实验 6|学习使用调度管理系统|
|实验 7|分布式训练任务练习|
|实验 8|自动机器学习系统练习|
|实验 9|强化学习系统练习|
||||

## 人工智能系统教材

[《人工智能系统》](https://github.com/microsoft/AI-System/tree/main/Textbook) 教材为[微软人工智能教育与共建社区](https://github.com/microsoft/ai-edu)中规划的人工智能相关教材之一。我们注意到，现在的大部分人工智能相关的教材，特别是深度学习和机器学习相关课程主要集中在相关理论、算法或者应用，与系统相关的教材并不多见。我们希望人工智能系统教材能让人工智能系统教育变得更加体系化和普适化，以共同促进人工智能与系统交叉人才的培养。

## 附录

\<TBD>


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Legal Notices

Microsoft and any contributors grant you a license to the Microsoft documentation and other content
in this repository under the [Creative Commons Attribution 4.0 International Public License](https://creativecommons.org/licenses/by/4.0/legalcode),
see the [LICENSE](LICENSE) file, and grant you a license to any code in the repository under the [MIT License](https://opensource.org/licenses/MIT), see the
[LICENSE-CODE](LICENSE-CODE) file.

Microsoft, Windows, Microsoft Azure and/or other Microsoft products and services referenced in the documentation
may be either trademarks or registered trademarks of Microsoft in the United States and/or other countries.
The licenses for this project do not grant you rights to use any Microsoft names, logos, or trademarks.
Microsoft's general trademark guidelines can be found at http://go.microsoft.com/fwlink/?LinkID=254653.

Privacy information can be found at https://privacy.microsoft.com/en-us/

Microsoft and any contributors reserve all other rights, whether under their respective copyrights, patents,
or trademarks, whether by implication, estoppel or otherwise.
