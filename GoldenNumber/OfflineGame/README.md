# 简介

离线的黄金点游戏要求玩家通过写机器人程序来参加黄金点游戏比赛。

- [Player目录](./Player)下提供玩家的机器人示例程序、模拟测试程序，玩家通过修改机器人产生预测值的逻辑来争取赢得比赛，具体使用方法参见[Player中的文档](./Player/README.md)

- [GameMaster目录](./GameMaster)下提供机器人的离线比赛环境，要求收集所有玩家的机器人程序到一台机器上，统一调度执行进行比赛，具体使用方法参见[GameMaster中的文档](./GameMaster/README.md)

# 什么是“黄金点”游戏？
请在邹欣老师的博客查看游戏介绍：[创新的时机 – 黄金点游戏](https://blog.csdn.net/SoftwareTeacher/article/details/25794525)

这里黄金点游戏比赛规则和上面邹欣老师博客中有稍许不同，允许每个玩家猜两个数

# 步骤

1.  将Player目录分发到各个玩家，玩家按照[Player中的文档](./Player/README.md)说明，借助模拟测试程序，实现并改进自己的机器人逻辑，最终产出机器人程序。
    > Tips: 每个玩家产出的机器人程序为一个文件夹，该文件夹下应包含一个可执行程序及其运行时所依赖的动态库等，或者该文件夹中包括一个get_numbers.py文件(如果是以python实现机器人)

2. 管理员收集所有玩家的机器人程序，每个玩家的机器人程序为一个文件夹，如`001`、`002`，然后所有玩家再放在同一文件夹中，如`bots`。示例如下：

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

3. 编译运行GameMaster目录下的机器人比赛环境LocalManager，按照[GameMaster中的文档](./GameMaster/README.md)说明，配置参加比赛的机器人及要进行的回合数进行比赛，并查看最终比赛结果