# 共建指南

## 贡献内容主要步骤
   1. Fork 此仓库；

   1. 根据您要贡献的内容，参考 [您可以贡献的内容](#您可以贡献的内容) 中的说明，在相应模块添加内容。

   1. 提交PR。如有必要，请附加详细的内容说明。

   我们总结了一些高效贡献的技巧，具体内容请参阅[如何高效贡献](#社区体验和代码贡献技巧)。

## 您可以贡献的内容

### 贡献人工智能基础或高级教程到[基础教程](../基础教程)
如果您想贡献新的人工智能教程到此模块，您需要确保所贡献内容是原创内容，建议您提前与我们沟通联系和讨论。您可以发送邮件到[opencode@microsoft.com](mailto:opencode@microsoft.com) 或 在GitHub上创建 [issue](https://github.com/Microsoft/ai-edu/issues)。

如果您想对已有教程进行修改或补充，您可以按照[贡献内容主要步骤](#贡献内容主要步骤)进行操作，并提交PR，我们会及时处理。

### 贡献案例和文档到[实践案例](../实践案例)
我们欢迎您贡献原创的案例和配套的文档、解决方案到模块[实践案例](../实践案例)。对于文档和图例，您可以将其直接添加到该模块中；对于配套的解决方案源代码，我们建议您新建一个专门的GitHub仓库用于放置和更新源代码，并将该仓库的链接粘贴到教案描述中。

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

1. 请在模块[实践案例](../实践案例)下新建一个案例文件夹，案例文件夹名应是您的案例名字（比如 `量化交易案例` ）。并在案例文件夹新建案例描述文件 `README.md`。

1. 在案例文件夹中新建一个解决方案文件夹，并为它设置一个易于分辨的名字，以便将您的解决方案和后续的其他解决方案区分开来（比如 `微软-方案1` ）。

1. 如果模块[实践案例](../实践案例)下已经存在同名的案例文件夹，您可以直接按照第2步，在该案例文件夹中添加新的解决方案文件夹。

对于案例文件夹，我们推荐您将您案例的主题作为文件夹的名字，前缀加上模块序号和案例序号。比如 `B11-量化交易案例` 或 `B04-智能家居`。

对于解决方案文件夹，我们推荐您使用 `机构/个人名称 + 方案技术要点` 的格式来为其命名。比如 `微软-基于Tools for AI`。

我们欢迎您同时提供与解决方案相对应的源代码。为了方便您更新和维护源代码，我们建议您创建一个专门的GitHub仓库用于放置和更新源代码，并将该仓库的链接粘贴到教案描述中。（关于如何创建新的GitHub仓库，请参考<https://help.github.com/en/articles/create-a-repo>。）

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

## 本社区注明版权出处的内容适用于[License](../LICENSE.md)版权许可。

# 社区体验和代码贡献技巧

本文档根据不同的使用需求，总结了一些社区资源的使用技巧，帮助您方便高效地体验本开源社区。

## 浏览社区文档

如果您的使用目的是仅浏览相关教程和文档，您可以直接**使用网页**浏览。

1. 本社区文档中，尤其是[基础教程](./基础教程)模块，有大量的公式。为了方便查看公式，可以使用Chrome浏览器，安装[Math展示控件](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima)。如有其他好的控件，欢迎通过[issues](https://github.com/microsoft/ai-edu/issues)模块反馈给我们。
2. 如需更好的文档阅读体验，可以clone该仓库，使用VSCode浏览，并安装**Markdown All in One**插件。

## 运行样例程序

如果您的使用目的除了阅读文档，还需运行教程内程序，请**clone**该开源仓库到本地。所有程序可通过VSCode或Visual Studio直接运行。您也可以修改您本地的仓库内容。

clone命令：
```
git clone https://github.com/microsoft/ai-edu.git
```

## 向社区贡献代码

如果您的使用目的包含向社区贡献代码，请**Fork**主分支的内容到您自己的GitHub账号下，之后所有的修改，基于您自己账号下的仓库进行。
具体步骤如下：
1. 登录您的GitHub账号，访问[microsoft/ai-edu](https://github.com/microsoft/ai-edu)社区。单击页面右上角的“Fork”按钮，fork本仓库到您自己的账号。
2. 在您自己的仓库中编辑修改内容。
3. 测试和审校您修改的内容，确保代码和文档的正确性。
4. 提交PR。
5. 通过审校的PR，可merge到源仓库中。

请特别注意，在您提交PR之前，请确保您已拉取远程仓库更新，并处理所有与源仓库的文档或代码冲突。相关命令如下:

1. 查看远程仓库状态
   ```
   git remote -v
   ```
2. 添加远程仓库（源仓库）
   ```
   git remote add upstream https://github.com/microsoft/ai-edu.git
   
   //upstream是为远程仓库起的别名，您可任意命名。
   ```
3. 拉取远程仓库
   ```
   git fetch upstream
   ```
4. 将远程仓库主分支的更新合并到本地主分支
   ```
   git checkout master
   git merge upstream/master
   ```
5. merge时如出现冲突，请解决冲突