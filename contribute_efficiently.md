# 社区体验和代码贡献技巧

本文档根据不同的使用需求，总结了一些社区资源的使用技巧，帮助您方便高效地体验本开源社区。

## 浏览社区文档

如果您的使用目的是仅浏览相关教程和文档，您可以直接**使用网页**浏览。

1. 本社区文档中，尤其是[A-基础教程](./A-基础教程)模块，有大量的公式。为了方便查看公式，可以使用Chrome浏览器，安装[Math展示控件](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima)。如有其他好的控件，欢迎通过[issues](https://github.com/microsoft/ai-edu/issues)模块反馈给我们。
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