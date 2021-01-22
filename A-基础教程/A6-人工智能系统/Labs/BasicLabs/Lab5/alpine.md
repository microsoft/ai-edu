## 运行你的第一个容器
当前章节假设你已经配置好Docker环境. 在本小节，你将在你的操作系统中运行一个 [Alpine Linux](http://www.alpinelinux.org/)容器(container)-一个轻量的Linux 发行版操作系统 ，并测试使用`docker run`命令。

在开始的时候,请在命令行执行以下命令，Pull Docker镜像(image):
```
$ docker pull alpine
```

> **注意:** 取决于你如何在你的系统中安装的Docker, 执行完上面的命令后，你可能看到`permission denied`错误 。尝试以下教程中的命令[verify your installation](https://docs.docker.com/engine/getstarted/step_one/#/step-3-verify-your-installation)去修正这个问题。 如果是在Linux下执行，可以在 `docker`命令前追加`sudo`. 或者可以[create a docker group](https://docs.docker.com/engine/installation/linux/ubuntulinux/#/create-a-docker-group)去解决这个问题。

`pull`命令会从从**Docker registry** 下载alpine 镜像(**image**)并在你的系统中保存。你可以使用 `docker images` 命令去罗列出当前你的系统中的所有镜像。
```
$ docker images
REPOSITORY              TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
alpine                 latest              c51f86c28340        4 weeks ago         1.109 MB
hello-world             latest              690ed74de00f        5 months ago        960 B
```

### Docker Run命令启动容器
让我们来基于当前镜像运行一个Docker **container**. 我们通过执行`docker run` 命令，运行容器.

```
$ docker run alpine ls -l
total 48
drwxr-xr-x    2 root     root          4096 Mar  2 16:20 bin
drwxr-xr-x    5 root     root           360 Mar 18 09:47 dev
drwxr-xr-x   13 root     root          4096 Mar 18 09:47 etc
drwxr-xr-x    2 root     root          4096 Mar  2 16:20 home
drwxr-xr-x    5 root     root          4096 Mar  2 16:20 lib
......
......
```

执行以上命令发生了什么？当你执行完`run`,会触发以下流程：
1. Docker 客户端(client)与Docker daemon通信
2. Docker daemon检查本地存储是否镜像在本地存在，在本实例中是(alpine in this case)。如果本地没有，从Docker Hub或Registry中下载镜像. (Since we have issued `docker pull alpine` before, the download step is not necessary)
3. Docker daemon会创建相应容器并在容器中执行一条命令。
4. Docker daemon将命令执行结果重定向到Docker client

当你运行`docker run alpine`, 由于你提供了一条Shell命令(`ls -l`), 所以Docker启动了相应命令，并且你看到了执行结果。

让我们尝试一下其他命令。

```
$ docker run alpine echo "hello from alpine"
hello from alpine
```

这是一些真实的输出。在这个实例中，Docker客户端在alpine容器中国执行了`echo`命令并之后退出。当前的执行相比虚拟机的启动，执行命令，并退出要快的多。 If you've noticed, all of that happened pretty quickly. Imagine booting up a virtual machine, running a command and then killing it. Now you know why they say containers are fast!

尝试另一条命令
```
$ docker run alpine /bin/sh
```

为何当前什么也没有发生？是由于Bug吗？其实是因为当执行完这些脚本后这些交互式的shell会退出。除非将命令运行与交互式的终端，例如本例子可以执行以下命令进入交互式模式`docker run -it alpine /bin/sh`。

当前你已经在容器的shell中，你可以尝试一些命令，例如`ls -l`, `uname -a` 或者其他命令。可以通过`exit`命令退出容器。


退出后，我们可能会使用`docker ps`命令，因为`docker ps` 命令会向用户展示所有在运行的容器和其信息。

```
$ docker ps
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
```

由于当前没有容器在运行，你只看到了一个空白行。让我们尝试一个更有用的命令配: `docker ps -a`

```
$ docker ps -a
CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS                      PORTS               NAMES
36171a5da744        alpine              "/bin/sh"                5 minutes ago       Exited (0) 2 minutes ago                        fervent_newton
a6a9d46d0b2f        alpine             "echo 'hello from alp"    6 minutes ago       Exited (0) 6 minutes ago                        lonely_kilby
ff0a5c3750b9        alpine             "ls -l"                   8 minutes ago       Exited (0) 8 minutes ago                        elated_ramanujan
c317d0a9e3d2        hello-world         "/hello"                 34 seconds ago      Exited (0) 12 minutes ago                       stupefied_mcclintock
```

以上展示的是你运行的所有容器。注意`STATUS`列提示这些容器刚刚在几分钟前退出。当前你可能想能否有一种方式能让我们在容器中执行更多的命令。我们可以通过以下命令做到：

```
$ docker run -it alpine /bin/sh
/ # ls
bin      dev      etc      home     lib      linuxrc  media    mnt      proc     root     run      sbin     sys      tmp      usr      var
/ # uname -a
Linux 97916e8cb5dc 4.4.27-moby #1 SMP Wed Oct 26 14:01:48 UTC 2016 x86_64 Linux
```
运行`run`命令时并附加`-it`选项，会让我们以一种交互式的方式进入容器中，这样用户可以执行更多所需要执行的命令。 

如果想查询`run`命令的更多用法, 用户可以是哟领`docker run --help`去查询它所有支持的选项列表。 

### 2.2 术语表

让我们在进行后面的教程前，先认识一些容易产生困惑但是在Docker生态系统中又比较重要的概念术语：

- *Images* - 应用的文件系统和配置，通过使用镜像才能创建容器。如果想查看镜像更多的信息，执行命令`docker inspect alpine`。在以上的实例中，使用 `docker pull`命令下载**alpine**镜像。当用户执行命令`docker run hello-world`, 它默认也会在后台执行`docker pull`去下载**hello-world** 镜像。
- *Containers* - 正在运行的Docker镜像实例。容器运行真正的应用。一个容器包含一个应用所包含的所有依赖，它与其他容器共享内核，并作为在主机操作系统中的一个被隔离的用户空间进程。用户可以通过`docker run`创建已下载的镜像的容器。用户可以通过`docker ps`查看当前正在运行的容器
- *Docker daemon* - 在主机上运行的后台服务，它管理构建，运行和销毁容器。
- *Docker client* - 允许用户与Docker daemon交互的命令行工具。
- *Docker Store* - 一个Docker镜像注册表[registry](https://store.docker.com/), 在这里用户可以找到可信的和企业级的容器镜像，插件，和Docker版本。

## 本小节作业

1. 按以上步骤创建一个alpine容器
2. 执行Linux监控命令，观察创建的容器的资源和命名空间和主机操作系统观察到的有何不同？

## 下一步
下一步教程 [2. Docker部署PyTorch训练程序](./train.md)
