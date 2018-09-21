# 离线黄金点游戏 -- 机器人比赛环境

## 简介

LocalManager提供了离线黄金点游戏的机器人比赛环境。

借助该环境，管理员可以调度执行玩家提交的机器人程序，并给机器人程序提供历史数据作为输入，在机器程序运行结束时读取输出的预测值，最后在每回合结束时计算黄金点及玩家的得分。该环境支持设定比赛回合数以及显示实时比赛结果。

![screenshot](./screenshot.png)

## 准备

* 将Player目录分发到各个玩家，玩家按照[Player中的文档](../Player/README.md)说明，借助模拟测试程序，实现并改进自己的机器人逻辑，最终产出机器人程序。
    > Tips: 每个玩家产出的机器人程序为一个文件夹，该文件夹下应包含一个可执行程序及其运行时所依赖的动态库等，或者该文件夹中包括一个get_numbers.py文件(如果是以python实现机器人)

* 管理员收集所有玩家的机器人程序，每个玩家的机器人程序为一个文件夹，如`001`、`002`，然后所有玩家再放在同一文件夹中，如`bots`。示例如下：

    ```
    \---bots
        +---001
        |   |   bot.exe
        |           
        +---002
        |   |   bot.exe
        |   |   depends files ...
        |           
        +---003
        |   |   get_numbers.py
        |           
        +---004
        |   |   get_numbers.py
        |   |   other files ...
    ```

    > 一些适合的程序的例子有：
    >
    > - [../Player/Bot/Bot.csproj](../Player/Bot/) 编译而产生的 `Bot.exe` 执行文件。可以用 Visual Studio 打开 [../Player/Player.sln](../Player/) 来进行编译。
    >
    > - ../Player/OfflineScoreboard/TestScript/get_numbers.py](../Player/OfflineScoreboard/TestScript/)。
    >
    > - 其他正确实现了数据输入、输出的可执行程序或脚本。输入、输出规范请参见 [../Player/README.md](../Player/README.md) 中相关章节。

    如果玩家的机器人最终产出是 exe 可执行程序，那么对应的目录下应包括该 exe 运行时所依赖的所有文件，保证该 exe 在干净环境中也可以运行（这些依赖文件一般是一些 dll 文件）。如果玩家的机器人是 python 脚本，**请务必命名为 `get_numbers.py`**。

* 运行比赛环境的机器上需要安装python，并配置对应的环境变量。还需要和玩家统一python版本号，如果玩家需要用到numpy或其它库，也需要提前配置好。

* 编译运行GameMaster目录下的机器人比赛环境LocalManager，具体操作请参见下一章节。

## 步骤

1. 编译运行LocalManager程序，即比赛环境。

2. 点击选择Bots所在目录，选择前面准备工作中收集的机器人的目录。

    > 日志窗口中应显示出发现了哪些Bot及其编号

3. 设定要运行的回合数，如400

4. 点击开始

    > 比赛过程中会实时显示比赛数据，可以看到每一回合的黄金点及走势，也可以看到玩家的得分情况

5. 比赛结束后，点击导出比赛数据，可以导出历史数据、玩家得分、日志

