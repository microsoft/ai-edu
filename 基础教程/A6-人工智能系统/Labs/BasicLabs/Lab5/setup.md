## 环境设置

### 1 前置依赖

推荐操作系统：Ubuntu 16.04 or 18.04

### 1.1 配置你的电脑或服务器

Docker的官方文档中的*getting started*有关于不同操作系统的Docker细节配置方法

* [Linux](https://docs.docker.com/engine/installation/linux/)
* [Mac](https://docs.docker.com/docker-for-mac/)
* [Windows](https://docs.docker.com/docker-for-windows/)

*如果你使用Docker for Windows* 请确保已经 [共享驱动](https://docs.docker.com/docker-for-windows/#shared-drives).

*注意事项* 如果你使用的是旧版本的Windows或者Mac系统，可能你需要使用[Docker Machine](https://docs.docker.com/machine/overview/)进行替代.

*以下的命令可以在bash或者 Powershell on Windows中执行*

如果你已经安装好Docker，通过以下命令测试你的环境已经安装成功:
```
$ docker run hello-world
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
03f4658f8b78: Pull complete
a3ed95caeb02: Pull complete
Digest: sha256:8be990ef2aeb16dbcb9271ddfe2610fa6658d13f6dfb8bc72074cc1ca36966a7
Status: Downloaded newer image for hello-world:latest

Hello from Docker.
This message shows that your installation appears to be working correctly.
...
```
## 下一步
点击进入本教程的下一步 [2. 运行你的第一个容器](alpine.md)
