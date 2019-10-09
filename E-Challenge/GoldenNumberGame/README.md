## 挑战黄金点

什么是黄金点游戏，请在邹欣老师的博客查看介绍：[创新的时机 – 黄金点游戏](https://blog.csdn.net/SoftwareTeacher/article/details/25794525)。

规则：N个玩家，每人写一个或两个0~100之间的有理数 (不包括0或100)，提交给服务器，服务器在当前回合结束时算出所有数字的平均值，然后乘以0.618（所谓黄金分割常数），得到G值。提交的数字最靠近G（取绝对值）的玩家得到N分，离G最远的玩家得到－2分，其他玩家得0分。只有一个玩家参与时不得分。

如何挑战：

* 人工挑战：打开[网页客户端](https://goldennumber.aiedu.msra.cn/)或者使用[本地客户端](../../C-开发工具与环境/微软黄金点程序工具/OnlineGame/SampleClient)，提交你的预测值，看能否最接近黄金点。
* 写AI来挑战：[用C#写个bot](../../C-开发工具与环境/微软黄金点程序工具/OnlineGame/BotDemoInCSharp)，或者[用Python写个bot](../../C-开发工具与环境/微软黄金点程序工具/OnlineGame/BotDemoInPython)，或者[用你熟悉的语言来写个bot](../../C-开发工具与环境/微软黄金点程序工具/OnlineGame)，让你的bot智能的选择数字进行提交，看能否在多回合比赛中得分最高。

比赛说明：

* 服务器接口详细说明及bot示例，请参考[这里](../../C-开发工具与环境/微软黄金点程序工具/OnlineGame)。
* [房间0](https://goldennumber.aiedu.msra.cn/main?roomid=0)和[房间1](https://goldennumber.aiedu.msra.cn/main?roomid=1)是常规比赛房间，其中内置了多个基于规则的bot会和你一起比赛。
* [房间0](https://goldennumber.aiedu.msra.cn/main?roomid=0)要求每个玩家每回合提交一个数，[房间1](https://goldennumber.aiedu.msra.cn/main?roomid=1)要求每个玩家每回合提交两个数，可以用两个数双保险去接近黄金点值，也可以提交一个较大的数字来保证自己的另一个数字比较接近黄金点值。
* 统计得分时，只累加当天进行的各回合比赛的得分。

如果您的得分最高，请在Issues里留言，并提供您的比赛日期、用户ID等信息，我们将进行评审，并邀请您的AI与其它智能的bot进行比赛。

另外，黄金点比赛正处在起步阶段，还有很多不足和考虑不周，您有任何建议都可以直接在本社区的Issues中留言。