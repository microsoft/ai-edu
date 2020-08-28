# 共建指南

## 贡献内容主要步骤
   1. Fork 此仓库；

   1. 根据您要贡献的内容，参考 [您可以贡献的内容](#您可以贡献的内容) 中的说明，在相应模块添加内容。

   1. 提交PR。如有必要，请附加详细的内容说明。

   我们总结了一些高效贡献的技巧，具体内容请参阅[如何高效贡献](./contribute_efficiently.md)。

## 您可以贡献的内容

### 贡献人工智能基础或高级教程到[A-基础教程](./A-基础教程)
如果您想贡献新的人工智能教程到此模块，您需要确保所贡献内容是原创内容，建议您提前与我们沟通联系和讨论。您可以发送邮件到[opencode@microsoft.com](mailto:opencode@microsoft.com) 或 在GitHub上创建 [issue](https://github.com/Microsoft/ai-edu/issues)。

如果您想对已有教程进行修改或补充，您可以按照[贡献内容主要步骤](#贡献内容主要步骤)进行操作，并提交PR，我们会及时处理。

### 贡献案例和文档到[B-实践案例](./B-实践案例)
我们欢迎您贡献原创的案例和配套的文档、解决方案到模块[B-实践案例](./B-实践案例)。对于文档和图例，您可以将其直接添加到该模块中；对于配套的解决方案源代码，我们建议您新建一个专门的GitHub仓库用于放置和更新源代码，并将该仓库的链接粘贴到教案描述中。

该模块的内容是按如下的文件夹结构来组织的：
```
B:教学案例与实践
├─案例1
│   ├─方案1
│   │   ├─子文件夹1
│   │   ├─子文件夹2
│   │   ├─其他文件1.jpg
│   │   ├─其他文件2.pdf
│   │   ├─其他文件3.md
│   │   └─README.md （方案描述文件）
│   ├─方案2
│   │   └─README.md （方案描述文件）
│   └─README.md （案例描述文件）
└─案例2
    ├─方案1
    │   └─README.md （方案描述文件）
    ├─方案2
    │   └─README.md （方案描述文件）
    └─README.md （案例描述文件）
```

您可以按照如下步骤添加您的案例和解决方案：

1. 请在模块[B-实践案例](./B-实践案例)下新建一个案例文件夹，案例文件夹名应是您的案例名字（比如 `量化交易案例` ）。并在案例文件夹新建案例描述文件 `README.md`。

1. 在案例文件夹中新建一个解决方案文件夹，并为它设置一个易于分辨的名字，以便将您的解决方案和后续的其他解决方案区分开来（比如 `微软-方案1` ）。

1. 如果模块[B-实践案例](./B-实践案例)下已经存在同名的案例文件夹，您可以直接按照第2步，在该案例文件夹中添加新的解决方案文件夹。

对于案例文件夹，我们推荐您将您案例的主题作为文件夹的名字，比如 `量化交易案例` 或 `语言理解应用案例 - 智能家具`。
如果您有多个属于同一系列（比如 `AI301系列`）的案例，我们推荐您使用系列名称作为案例文件夹名字的前缀，比如 `AI301 - 量化交易案例`。

对于解决方案文件夹，我们推荐您使用 `机构/个人名称 + 方案技术要点` 的格式来为其命名。比如 `微软-基于Tools for AI`。

我们欢迎您同时提供与解决方案相对应的源代码。为了方便您更新和维护源代码，我们建议您创建一个专门的GitHub仓库用于放置和更新源代码，并将该仓库的链接粘贴到教案描述中。（关于如何创建新的GitHub仓库，请参考<https://help.github.com/en/articles/create-a-repo>。）

### 贡献到[C-挑战项目](./C-挑战项目)
贡献到[C-挑战项目](./C-挑战项目)的内容主要为人工智能应用开发中需要共建者贡献代码解决的挑战及作业。贡献的内容包括：
   - 人工智能教学者，可发布作业及习题（需要清楚的习题描述）。
   - 人工智能应用开发者、学习者，可发布需要应战者贡献代码解决的问题（需要清楚的问题描述）。
   - 应战者可以贡献代码和说明文档，展示如何解决发布的相应的问题。
   - 请新建Markdown文件用作所发布问题的描述文件。

### 贡献人工智能应用开发工具、环境搭建教程到[D-工具环境](./D-工具环境)
   - 如果是开发工具，提供工具安装包或者相应案例。
   - 推荐提供开发工具配套的环境搭建教程。

### 贡献人工智能教学课程大纲或者课件到[E-课程集锦](./E-课程集锦)
   - 确保上传的文本为 Markdown 格式。
   - 如果您贡献的课程为PPT格式，请转换为PDF格式，上传至您自己的GitHub仓库。并在本社区添加课件简介和仓库链接。欢迎使用**GitHub page**展示教学资源。具体内容请参阅[如何使用GitHub Pages展示教学资源](./E-课程集锦/如何使用GitHub%20Pages展示教学资源.md)。
   - 可以贡献一个新的原创人工智能相关的教学大纲、课件。
   - 也可以贡献基于已存在的教学大纲、案例进行改进、定制后的教学大纲、课件。比如从本社区的资源选取相应内容重组成一个适应不同专业、不同难度的新大纲或课件。
   - 推荐在贡献的内容中（大纲、课件）内容中说明课程适用对象、课程难度、建议课时等信息。

### 贡献到[F:答疑交流](./F-答疑交流)
贡献到[D:答疑与交流](./D-答疑与交流)的内容可以为人工智能学习中，您：
   - 遇到的问题，也可以直接在“issues”tab里提问。
   - 对别人问题的解答，也可以直接在“issues”tab里回答。
   - 学到的心得。
   - 总结的FAQ。

## 在提交Pull request之前，请确保：
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
  
## Contributor License Agreement

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## 本社区注明版权出处的内容适用于[License](./LICENSE.md)版权许可。