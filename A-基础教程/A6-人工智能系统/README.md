# 人工智能系统（System for AI）教程说明

## 如何使用本教程

本教程的内容以 `git submudule` 的形式加入**人工智能教育与学习共建社区**。本教程的原始链接为 [Microsoft/AI-System](https://github.com/microsoft/AI-System)。您可以通过如下两种方式获取教程内容。

### 访问并克隆 AI-System 仓库

仓库地址： https://github.com/microsoft/AI-System

### 在本仓库（AI-Edu）中添加 submodule

1. 如果您还未克隆 AI-Edu 仓库，可使用如下命令克隆整个仓库。
   ```
   git clone --recursive https://github.com/microsoft/ai-edu.git
   ```
   使用 `--recursive` 参数，可将submodule一同clone下来。

2. 如果您已经克隆 AI-Edu 仓库，并未包含此教程模块，可使用如下命令添加此模块。
   ```
   git submodule init
   ```

3. 如果您要拉取此模块最新内容，请使用如下命令。
   ```
   git submodule update
   ```
   或者您可使用如下命令完成上面两步：
   ```
   git submodule update --init
   ```

更多关于submodule的使用方式，请参考如下链接：

- [Git Submodules basic explanation](https://gist.github.com/gitaarik/8735255)
- [Git 官方文档](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [Working with submodules](https://github.blog/2016-02-01-working-with-submodules/)

## 本教程内容简介

请阅读 [人工智能系统中文文档](https://github.com/microsoft/AI-System/blob/main/README_cn.md) 获取课程介绍。