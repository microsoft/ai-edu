# Code Search

## 介绍

近年来，人工智能逐渐进入各个领域并展现出了强大的能力。在计算机视觉领域，以imagenet为例，计算机的图像分类水平已经超过了人类。在自然语言处理(NLP)领域，BERT、XLNet以及MASS也一遍遍的刷新着任务榜单。当人工智能进入游戏领域，也取得了惊人的成绩，在Atari系列游戏中，计算机很容易超过了大部分人类，在围棋比赛中，Alpha Go和Alpha Zero也已经超越了人类顶尖棋手。

程序代码是一系列的语法标记，看起来很像人类自然语言编写的文本。如果人工智能进入PL/SE领域，可以带来什么改变呢？

随着近年来NLP在拼写检查、机器翻译、语义分析和问答方面的快速发展及广泛应用，人们期望可以从大型代码库（如GitHub）中发现类似于自然语言的模式，并利用NLP的相关技术来进行解决。但是，程序代码和人类文本又有一些区别，比如程序代码是可执行的，在语义上常常很脆弱，很小的改动就能彻底改变代码的含义，而自然语言中如果出现个别错误可能也不会影响人们的理解。这些差异给我们直接应用现有的NLP技术带来了挑战，因此需要一些新的方法来处理程序代码。

人工智能在PL/SE领域可以有很多的挑战，如语义搜索、代码补全（完形填空）、自动生成单元测试、代码翻译、代码缺陷检测、代码转文本、自然语言生成代码等。这里面有的挑战是很有难度的，如自然语言自动产生代码，需要理解自然语言并按要求产生可用的代码，但是我们可以尝试先从简单的任务开始做些挑战。

这里我们选择代码搜索作为任务，使得用户可以使用自然语言进行查询，然后在数据集中找到符合用户需求的代码片段返回给用户。

程序员在遇到不会的问题或者不熟悉的语言时，通常会在网上查找相关的代码来参考。但是传统的代码搜索多数是基于关键字匹配的，比如直接在GitHub上进很搜索，通常需要知道要搜索的代码中包含什么，然后进行搜索，而直接使用自然语言进行查询的结果并不理想。再比如StackOverflow，虽然可以使用自然语言进行查询，但问题的相关解答中可能并不包含对应的代码信息，而通常情况下程序员更想直接看到最相关的程序代码。如果我们可以解决这个问题，那就可以极大的优化代码搜索的体验。

## 目标

本次实习的任务是在网上收集数据并进行预处理，搭建模型并实现使用自然语言搜索代码的功能，最终完成一个VSCode扩展可以展示这个功能。

数据可以在GitHub、StackOverflow或其它网站爬取，也可以下载已公开的数据集。比如可以在GH Archive上查询出满足要求的GitHub Repo，然后再爬取对应的数据。注意，爬取数据时一定要注意对应的开源协议。

数据的预处理也非常的重要。原始数据通常不直接做为模型的输入，尤其是程序代码，通常情况下一些适当的变换和信息抽取是必需的，可以参考后面列出的一些资料中的处理方式。

模型是这个任务最核心的部分，后面的参考资料中分别介绍了几种模型和思路，有几种模型的思路是类似的，同学们可以研究一下。[How To Create Natural Language Semantic Search For Arbitrary Objects With Deep Learning](https://towardsdatascience.com/semantic-code-search-3cd6d244a39c)中的示例教程提供了一个步骤非常详细并且可以使用的方案，同学们一开始可以尝试下这个模型。在研究了所有资料后，同学们可以提出自己的模型或者对个模型进行改进，并完成新模型的训练。最后可以评估一下你的模型能比这个示例教程好多少，后面的参考论文中有几篇提到了评估的方法，可以进行参考。

最终的VSCode扩展不用做的很复杂，添加一个command支持搜索，并能显示出最匹配的几条代码片段即可，同时需要记录用户的查询及最终选择的代码片段，辅助进行模型的评估及改进。

Code Search的同学分成了两个组，建议第一组同学实现Python语言的代码搜索，第二组同学实现JavaScript语言的代码搜索。两组同学之间不是竞争的关系，而且有很多关键技术是重合的，希望两组同学之间多进行交流合作，比如可以交流在哪可以找到更合适的数据集、对于论文中的模型有什么心得等。

## 具体安排

### 第一周

* 在Github 上注册自己的账号，学会基本操作，学会 Markdown 的文章写作。把账户地址告诉我们。 
* 通读下面所有文章，每一个文章列出至少一个你还没能弄懂的问题。每一个同学要把问题写在自己的github 账户中的一个 markdown 文档里面。 请看这个文章： https://www.cnblogs.com/xinz/p/9660404.html  里面的 “提有质量的问题” 部分。 把提问的文档地址告诉我们。
* 读完文章后， 如果你来做有创新性的  Code Search 实用工具， 你能提出什么想法？请按这里的 NABCD 模板提出想法，其中，请看这里的介绍，用里面提到的方法去采访你们同学中用Python、JavaScript 的用户，了解他们在写程序过程中的痛点。 
* 阅读敏捷项目的管理（https://www.cnblogs.com/xinz/archive/2012/10/05/2712602.html ），我们会用类似的方法来管理项目，同学们自己先做好准备
* 完成以上内容的基础上，尝试将[How To Create Natural Language Semantic Search For Arbitrary Objects With Deep Learning](https://towardsdatascience.com/semantic-code-search-3cd6d244a39c)中的notebook教程跑一遍（原数据集太大，可在不考虑准确率的情况下，用极小的数据集跑通一遍）
* 将以上两个文档的链接回复在[Issue #282](https://github.com/microsoft/ai-edu/issues/282)中。
* 有任何问题也可以在[Issue #282](https://github.com/microsoft/ai-edu/issues/282)中提出。

### 第二周

交流学习并讨论并制定未来几周的计划，明确要做什么，最终产出什么。

### 第三、四周

10天sprint，每天scrum，报告进度

### 第五周

准备及进行项目审查

## 实习成果评估

各组除了完成项目要求外，还需要提供一份报告，报告中应对现有技术做一些总结和综述，然后介绍自己的模型，或者介绍有哪些改进，并说明是如何做的，最后还要对模型进行评估总结，指明新的模型是否更好，或者还有哪些可以改进的地方。

## 参考资料

* [A Survey of Machine Learning for Big Code and Naturalness](https://arxiv.org/pdf/1709.06182.pdf)
* [How To Create Natural Language Semantic Search For Arbitrary Objects With Deep Learning](https://towardsdatascience.com/semantic-code-search-3cd6d244a39c)
  * https://github.com/hamelsmu/code_search
* [Deep API Learning](https://guxd.github.io/papers/deepapi.pdf)
  * https://github.com/guxd/deepAPI
* [Deep Code Search](https://guxd.github.io/papers/deepcs.pdf)
  * https://github.com/guxd/deep-code-search
* [code2vec: Learning Distributed Representations of Code](https://arxiv.org/pdf/1803.09473.pdf)
	* https://code2vec.org/
* [Aroma: Code Recommendation via Structural Code Search](https://arxiv.org/pdf/1812.01158.pdf)
* [When Deep Learning Met Code Search](https://arxiv.org/pdf/1905.03813.pdf)
* [Maybe Deep Neural Networks are the Best Choice for Modeling Source Code](https://arxiv.org/pdf/1903.05734.pdf)
* [Visual Studio Code Extension API](https://code.visualstudio.com/api)