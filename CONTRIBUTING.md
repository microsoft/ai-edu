# 共建指南

# 在提交Pull request之前，请确保：
   - 您提交的内容不是已存在的共建内容。
   - 您提交的内容位于正确的模块、路径下。
   - 检查您的语法和书写。
   - 欢迎您对已存在的共建内容进行改进。
   - 提交Pull request时候请附带清楚的描述标题。
   - 如果现有的模块不能覆盖您想贡献的内容，欢迎创建新的模块。
   - 每一个案例或者教程，对应一个单独的Pull request。
   - 提交一条新的contribution内容，确保格式为 <pre><code>[Title] (link_to_tutorial)</code></pre> 如果是系列，格式为
     <pre><code>* Title
        * [Part 1](link_to_part_1)
        * [Part 2](link_to_part_2)</code></pre>
   - 好好地写commit信息, 仔细阅读每一个CONTRIBUTING文件。 
   - 不要到处留无用的空格。
   - 推荐书写相对链接实现站内跳转

# 您可以贡献的内容可以包括：
   - 贡献到[A:教学课程](./A-教学课程/README.md)的人工智能教学课程大纲。
       - 确保为markdown格式，文件格式如下：
       <pre><code>[Title+author] (link_to_tutorial)</code></pre> 
       - 可以是一个新的人工智能相关的教学大纲、课件。
       - 也可以是基于已存在的教学大纲、案例进行改进、定制后，分享改进后的教学大纲。比如从[B:教学案例与实践](./B-教学案例与实践/README.md)选取相应的案例重组成一个适应不同专业、不同难度的新大纲。
       - 推荐在贡献的大纲、课件内容中说明适用对象、课程难度、建议课时等信息。
  
   - 贡献到[B:教学案例与实践](./B-教学案例与实践/README.md)的可以为案例、教程、代码。
       - 可以是一个新的案例，新建一个文件夹，包含案例说明和参考的code。
       - 可以是对已经存在的案例，贡献不同的解决方案。如在 [自构建－图像识别应用案例-手写算式计算器](./B-学习资源/B9-自构建－图像识别应用案例-手写算式计算器/README.md) 案例模块中，可以为手写算式计算器添加其他的实现方案。格式举例如下，N为顺序号：
         * 在 [自构建－图像识别应用案例-手写算式计算器](./B-学习资源/B9-自构建－图像识别应用案例-手写算式计算器/README.md)模块下添加新的文件夹，命名格式为：<pre><code>[author+solutionN+title]</code></pre> n为该案例模块下的顺序第n个解决方案
         * 在 [自构建－图像识别应用案例-手写算式计算器](./B-学习资源/BB9-自构建－图像识别应用案例-手写算式计算器/README.md) 模块下添加新的markdown文件描述改实现方案，文件名命名格式如下，N为顺序号：<pre><code> READMEN+title</code></pre>
   
   - 贡献到[C:开发工具与环境](./C-开发工具与环境/README.md)的可以为人工智能应用开发工具，环境搭建教程等。
       - 如果是开发工具，提供工具安装包或者相应案例。
       - 推荐提供开发工具配套的环境搭建教程。
    
   - 贡献到[D:答疑与交流](./D-答疑与交流/README.md)的可以为人工智能学习中，你：
      - 遇到的问题，也可以直接在“issues”里提问。
      - 对别人问题的解答，也可以直接在“issues”里回答。
      - 学到的心得。
      - 总结的FAQ
  
   - 贡献到[E:等你来战](./E-等你来战)的主要为你提供的对于改模块下为解决的人工智能问题的解决方案。



# 仔细阅读以下声明：

# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Licenses

Some open-source code and materials are used. They are under different licenses:

- The script [mnist_extension.py](./AI301/self-built_mnist_extension/tensorflow_model/mnist_extension.py) is under [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0). Based on [Origin code](https://github.com/tensorflow/models/blob/f81bb397efe57cf8bfb4a195c1b3064997f3e3c2/tutorials/image/mnist/convolutional.py).
