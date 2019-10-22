# Code Search

## 介绍

近年来，人工智能逐渐进入各个领域并展现出了强大的能力。在计算机视觉领域，以imagenet为例，计算机的图像分类水平已经超过了人类。在自然语言处理(NLP)领域，BERT、XLNet以及MASS也一遍遍的刷新着任务榜单。当人工智能进入游戏领域，也取得了惊人的成绩，在Atari系列游戏中，计算机很容易超过了大部分人类，在围棋比赛中，Alpha Go和Alpha Zero也已经超越了人类顶尖棋手。

随着近年来人工智能的自然语言处理（Natural Language Processing, NLP）技术在拼写检查、机器翻译、语义分析和问答方面的快速发展及广泛应用，人们期望可以从大型代码库（如GitHub）中发现类似于自然语言的模式，并利用相关技术来辅助程序员高效地编写无错代码。程序代码是一系列的语法记号，虽然看起来很像人类自然语言编写的文本，但是和人类文本有一些显著区别。比如程序代码是可执行的，在语义上常常很脆弱，很小的改动就能彻底改变代码的含义，而自然语言中如果出现个别错误可能也不会影响人们的理解。这些差异给我们直接应用现有的NLP技术带来了挑战，因此需要一些新的方法来处理程序代码。

人工智能在程序语言（Programming Language，PL）/软件工程（Software Engineering ，SE)领域可以有很多现实的应用，如语义搜索、代码补全（完形填空）、自动生成单元测试、代码翻译、代码缺陷检测、代码转文本、自然语言生成代码等。这里面有的方向是很有难度的，如自然语言自动产生代码，需要理解自然语言并按要求产生可用的代码，但是我们可以尝试先从简单的任务开始做些挑战。

我们选择“代码搜索”作为本次任务：接收用户的自然语言查询语句，在预定义代码集中找到符合用户需求的代码片段返回给用户。这个任务非常有意义。程序员在遇到不熟悉的编程语言或问题时，通常在网上查询相关的代码作为参考。但是传统的代码搜索多数是基于关键字匹配的，比如直接在GitHub上进很搜索，通常需要知道要搜索的代码中包含什么，然后才能进行搜索。直接使用自然语言进行查询并不能得到理想的结果。再比如StackOverflow，虽然可以使用自然语言进行查询，但问题的相关解答中可能并不包含程序员所需的代码信息。如果我们改进这个代码搜索的问题，那就可以极大地提高程序员的生产效率。


## 目标

本次实习的任务是在网上收集代码数据并进行预处理，搭建模型并实现使用自然语言搜索代码的功能，最终需要完成一个[Visual Studio Code](https://code.visualstudio.com/)（VS Code）扩展来展示功能。

数据可以在GitHub、StackOverflow或其它网站爬取，也可以下载已公开的数据集。比如可以在[GH Archive](https://www.gharchive.org/#bigquery)上查询出满足要求的GitHub Repo，然后再爬取对应的数据。注意，爬取数据时一定要注意代码的开源协议以及合理使用网络带宽。

数据的预处理是非常的重要。原始数据，尤其是程序代码，通常不能直接做为模型的输入，需要进行适当的变换和信息抽取。具体方法可以参考后面列出的一些资料中的处理方式。

模型是这个任务最核心的部分，后面的参考资料分别介绍了几种模型和思路。[How To Create Natural Language Semantic Search For Arbitrary Objects With Deep Learning](https://towardsdatascience.com/semantic-code-search-3cd6d244a39c)的示例教程提供了一个步骤非常详细并且可以使用的方案，同学们一开始可以尝试下这个模型。在研究了所有资料后，同学们可以提出自己的模型或者对某个参考模型进行改进，并完成训练。最后可以评估一下你的模型能比这个示例教程好多少（参考文献中有相关的评估方法）。

最终的VS Code扩展不用做的很复杂，添加一个command支持搜索，并能显示出最匹配的几条代码片段即可。同时需要记录用户的查询语句及最终选择的代码片段，便于后续的模型评估及改进。

## 具体安排

Code Search的同学分成了两个组，建议第一组同学实现Python语言的代码搜索，第二组同学实现JavaScript语言的代码搜索。两组同学之间不是竞争的关系，而且有很多关键技术是重合的，希望两组同学之间多进行交流合作，比如可以交流在哪可以找到更合适的数据集、对于论文中的模型有什么心得等。

### 第一周

* 在Github 上注册自己的账号，学会基本操作，学会 Markdown 的文章写作。把账户地址告诉我们。 
* 通读下面参考资料中`ML in PL/SE`和`Code Search Related`中的所有文章，每一个文章列出至少一个你还没能弄懂的问题。每一个同学要把问题写在自己的github 账户中的一个 markdown 文档里面。 请看这个文章： https://www.cnblogs.com/xinz/p/9660404.html  里面的 “提有质量的问题” 部分。 把提问的文档地址告诉我们。
* 读完文章后， 如果你来做有创新性的  Code Search 实用工具， 你能提出什么想法？请按这里的 [NABCD](https://www.cnblogs.com/xinz/archive/2010/12/01/1893323.html) 模板提出想法，其中，请看[这里的介绍](https://www.cnblogs.com/xinz/archive/2013/02/03/2890786.html)，用里面提到的方法去采访你们同学中用Python、JavaScript 的用户，了解他们在写程序过程中的痛点。
* 将以上GitHub链接回复在[Issue #282](https://github.com/microsoft/ai-edu/issues/282)中。
* 有任何问题也可以在[Issue #282](https://github.com/microsoft/ai-edu/issues/282)中提出。
* 阅读敏捷项目的管理（https://www.cnblogs.com/xinz/archive/2012/10/05/2712602.html ），我们会用类似的方法来管理项目，同学们自己先做好准备
* 完成以上内容的基础上，尝试将[How To Create Natural Language Semantic Search For Arbitrary Objects With Deep Learning](https://towardsdatascience.com/semantic-code-search-3cd6d244a39c)中的notebook教程跑一遍（原数据集太大，可在不考虑准确率的情况下，用极小的数据集跑通一遍）

### 第二周

交流学习并讨论制定未来几周的计划，明确要做什么，最终产出什么。创建给每一个团队成员的任务，确保每一个任务的估计时间不多于 6 小时（一天的工作时间）。
任务的基本格式要求：
* 任务标题，任务描述，任务的估计时间（小时），任务已经消耗了多少时间（小时），任务还剩余多少时间（小时）

### 第三、四周

10天sprint，每天scrum，报告进度，并用燃尽图的方法来汇报项目进展。 
每个人每天更新自己的任务 （task/issue）：已完成了多少小时的工作， 还剩余多少小时的工作
我们期待每人每天完成 6 小时的工作，一个工作的剩余时间可以比估计的时间增加。 另外，如果发现需要做新的工作，要创造新的任务（task）来跟踪。 
团队的PM 要把所有人的状态汇总显示为燃尽图。  图中按天数显示：迄今为止，已经完成的所有工作时间，目前所有任务的剩余的时间总和。 
请看网上关于SCRUM, 燃尽图的要求：
https://www.cnblogs.com/xinz/p/10011637.html 
https://www.cnblogs.com/xinz/archive/2012/10/05/2712602.html 


### 第五周

准备及进行项目审查，不用写PowerPoint，直接显示软件的运行效果，和工作的燃尽图等具体的记录。 

## 实习成果评估

各组除了完成项目要求外，还需要提供一份报告，报告中应对现有技术做一些总结和综述，然后介绍自己的模型，或者介绍有哪些改进，并说明是如何做的，最后还要对模型进行评估总结，指明新的模型是否更好，或者还有哪些可以改进的地方。

## 参考资料

* ML in PL/SE
  * [A Survey of Machine Learning for Big Code and Naturalness](https://arxiv.org/pdf/1709.06182.pdf)
  * [code2vec: Learning Distributed Representations of Code](https://arxiv.org/pdf/1803.09473.pdf)
	  * https://code2vec.org/
  * [Maybe Deep Neural Networks are the Best Choice for Modeling Source Code](https://arxiv.org/pdf/1903.05734.pdf)
* Code Search Related
  * [How To Create Natural Language Semantic Search For Arbitrary Objects With Deep Learning](https://towardsdatascience.com/semantic-code-search-3cd6d244a39c)
    * https://github.com/hamelsmu/code_search
  * [Deep API Learning](https://guxd.github.io/papers/deepapi.pdf)
    * https://github.com/guxd/deepAPI
  * [Deep Code Search](https://guxd.github.io/papers/deepcs.pdf)
    * https://github.com/guxd/deep-code-search
  * [Aroma: Code Recommendation via Structural Code Search](https://arxiv.org/pdf/1812.01158.pdf)
  * [When Deep Learning Met Code Search](https://arxiv.org/pdf/1905.03813.pdf)
  * [Introducing the CodeSearchNet challenge](https://github.blog/2019-09-26-introducing-the-codesearchnet-challenge/)
    * https://arxiv.org/abs/1909.09436
    * https://github.com/github/CodeSearchNet
    * https://app.wandb.ai/github/codesearchnet/benchmark
  * [Releasing a new benchmark and data set for evaluating neural code search models](https://ai.facebook.com/blog/neural-code-search-evaluation-dataset/)
    * https://arxiv.org/abs/1908.09804
    * https://github.com/facebookresearch/Neural-Code-Search-Evaluation-Dataset
* Visual Studio Code Extension Development
  * [Visual Studio Code Extension API](https://code.visualstudio.com/api)
