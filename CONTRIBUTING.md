# 共建指南
# 贡献内容主要步骤：
   - 第一步：根据您要贡献的内容，在相应模块添加内容，如有必要，请附加详细的说明文档。
   - 第二步：更新[readme](https://github.com/Microsoft/ai-edu/blob/master/README.md)文件，在该文件的目录结构添加指向您贡献的内容的链接。提交一条新的内容，确保格式为 <pre><code>[Title] (link_to_tutorial)</code></pre> 如果是系列，格式为
     <pre><code>* Title
        * [Part 1](link_to_part_1)
        * [Part 2](link_to_part_2)</code></pre>
   - 第三步：提交PR。

# 您可以贡献的内容可以包括：
   - 贡献到[A:教学课程](./A-教学课程/README.md)的人工智能教学课程大纲或者课件。
       - 确保上传的文本为markdown格式，添加到[readme](https://github.com/Microsoft/ai-edu/blob/master/README.md)的“目录结构”的文件标题格式如下：
       <pre><code>[Title+author] (link_to_tutorial)</code></pre> 例如：
        <pre><code>[人工智能实践课程大纲-微软System团队] (link_to_tutorial)</code></pre>
       - 可以贡献一个新的原创人工智能相关的教学大纲、课件。
       - 也可以贡献基于已存在的教学大纲、案例进行改进、定制后的教学大纲、课件。比如从本社区的资源选取相应内容重组成一个适应不同专业、不同难度的新大纲或课件。
       - 推荐在贡献的内容中（大纲、课件）内容中说明课程适用对象、课程难度、建议课时等信息。
  
   - 贡献到[B:教学案例与实践](./B-教学案例与实践/README.md)的可以为案例、代码。
       - 可以从此模块选取相应的案例进行教学。
       - 可以贡献原创的案例。
       - 如果贡献一个新的案例，请在模块B下新建一个文件夹，包含案例说明文件和相应code。
       - 如果贡献的内容是针对对已经存在的案例不同的解决方案，请在该案例下新建一个文件夹包含您的解决方案。比如：在 [自构建－图像识别应用案例-手写算式计算器](./B-学习资源/B9-自构建－图像识别应用案例-手写算式计算器/README.md) 案例模块中，想为手写算式计算器添加另外的实现方案，您可以遵循的格式举例如下：
         * 在 [自构建－图像识别应用案例-手写算式计算器](./B-学习资源/B9-自构建－图像识别应用案例-手写算式计算器/README.md)模块下新建一个文件夹，文件夹命名格式为：<pre><code>[title+solutionN+author]</code></pre> 这里的N为该案例模块下的顺序第n个解决方案。例如：
           <pre><code>[手写算式计算器-方案2-邹欣]</code></pre>
         * 在 [自构建－图像识别应用案例-手写算式计算器](./B-学习资源/BB9-自构建－图像识别应用案例-手写算式计算器/README.md) 模块下新建的文件夹内添加新的markdown文件描述该实现方案，文件名命名格式如下，N为顺序号，比如使用如下的文件名：<pre><code> 手写算式计算器方案2</code></pre>
   
   - 贡献到[C:开发工具与环境](./C-开发工具与环境/README.md)的可以为人工智能应用开发工具，环境搭建教程等。
       - 如果是开发工具，提供工具安装包或者相应案例。
       - 推荐提供开发工具配套的环境搭建教程。
    
   - 贡献到[D:答疑与交流](./D-答疑与交流/README.md)的可以为人工智能学习中，您：
      - 遇到的问题，也可以直接在“issues”tab里提问。
      - 对别人问题的解答，也可以直接在“issues”tab里回答。
      - 学到的心得。
      - 总结的FAQ。
  
   - 贡献到[E:等你来战](./E-Challenge/README.md)的内容主要为人工智能应用开发中需要共建者贡献代码解决的挑战及作业。贡献的内容包括：
      - 人工智能教学者，可发布作业及习题（需要清楚的习题描述）。
      - 人工智能应用开发者、学习者，可发布需要应战者贡献代码解决的问题（需要清楚的问题描述）。
      - 应战者可以贡献代码和说明文档，展示如何解决发布的相应的问题。
      - 可以新建markdown文件用于所发布问题的描述文件，也可以直接在此[Readme](./E-Challenge/README.md)里贡献内容。

# 在提交Pull request之前，请确保：
   - 如您的贡献内容引用了本社区的原创内容，请声明并注明出处。
   - 您提交的内容不是重复的内容。
   - 您提交的内容位于正确的模块、路径下。
   - 检查您的语法和书写。
   - 提交Pull request时候请附带清楚的描述标题。
   - 好好地写commit信息, 仔细阅读每一个CONTRIBUTING文件。 
   - 不要到处留无用的空格。
   - 如果现有的模块不能覆盖您想贡献的内容，欢迎创建新的模块。
   - 欢迎您对已存在的共建内容进行改进。
   - 确保书写的链接能实现正常跳转。
  



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

# 本社区注明版权出处的内容适用于[License](./LICENSE.md)版权许可。