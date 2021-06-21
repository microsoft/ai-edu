# Docker部署PyTorch训练程序

用户使用Docker部署应用程序时，一般要先写Dockerfile，通过Dockerfile构建镜像，通过镜像启动容器。
我们通过以下实例展开以上流程。

## 构建Dockerfile

下面内容为我们所创建的Dockerfile中的内容，Docker会按顺序通过Dockerfile在构建阶段会执行了一系列Docker命令(FROM, RUN, WORKDIR, COPY, EXPOSE, CMD)和Linux Shell命令，我们通过注释介绍相应的命令功能。

```sh
# 继承自哪个基础镜像
FROM nvidia/cuda:10.1-cudnn7-devel

# 创建镜像中的文件夹，用于存储新的代码或文件
RUN mkdir -p /src/app

# WORKDIR指令设置Dockerfile中的任何RUN，CMD，ENTRPOINT，COPY和ADD指令的工作目录
WORKDIR /src/app

# 拷贝本地文件到Docker镜像中相应目录
COPY pytorch_mnist_basic.py /src/app

# 需要安装的依赖

RUN apt-get update && apt-get install wget -y
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda
ENV PATH /opt/conda/bin:$PATH

RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# 容器启动命令
CMD [ "python", "pytorch_mnist_basic.py" ]

```

- 当Docker文件准备好，可以通过命令 `docker build -t train_dl .`构建镜像

---

**注意：` . `在`docker build`命令中是一个相对文件路径，指向你存储Dockerfile的位置，所以使用前请先确认已经进入了含有Dockerfile的文件夹***

---

当构建成功，会在控制台看到下面的日志
```sh
Successfully built 7e69d61fd488
```

## 构建Docker镜像

- 经过上面的操作，你已经具备构建了一个打包你的应用代码的镜像的前置国祚。接下来我们通过下面的命令去构建Docker镜像：

```sh
docker build -f Dockerfile.cpu -t train_dl .
```

`docker build` : 通过Dockerfile进而构建镜像的命令。

` -t` : 't'代表标签。用户通过标签未来可以确定相应的镜像。

`train_dl` : 给镜像打上的标签名字。

` . ` : 希望构建进镜像中的Dockerfile的相对路径。

- 用户可以通过命令`docker images`查看本地镜像

---

如果当前没有本地缓存 'pytorch/pytorch:latest'镜像, Docker会在构建阶段从Dockerhub Pull相应镜像到本地。

---

执行成功，会看到日志：

```
Successfully built 384b59932a42
Successfully tagged train_dl:latest
```

## 启动所构建镜像的容器实例

- 当前你的代码已经打包进镜像，并保存在本地机器。让我们通过下面的命令尝试启动它 `docker run --name training --rm train_dl`

成功会观察到日志：

```
...
Train Epoch: 1 [49920/60000 (83%)]      Loss: 0.195172
Train Epoch: 1 [50560/60000 (84%)]      Loss: 0.052903
Train Epoch: 1 [51200/60000 (85%)]      Loss: 0.262594
Train Epoch: 1 [51840/60000 (86%)]      Loss: 0.066995
Train Epoch: 1 [52480/60000 (87%)]      Loss: 0.022450
Train Epoch: 1 [53120/60000 (88%)]      Loss: 0.239718
Train Epoch: 1 [53760/60000 (90%)]      Loss: 0.152097
Train Epoch: 1 [54400/60000 (91%)]      Loss: 0.047996
Train Epoch: 1 [55040/60000 (92%)]      Loss: 0.250904
Train Epoch: 1 [55680/60000 (93%)]      Loss: 0.048776
Train Epoch: 1 [56320/60000 (94%)]      Loss: 0.090460
Train Epoch: 1 [56960/60000 (95%)]      Loss: 0.088897
Train Epoch: 1 [57600/60000 (96%)]      Loss: 0.124534
Train Epoch: 1 [58240/60000 (97%)]      Loss: 0.144476
...
```

- 执行`docker ps`检查是否当前容器已经启动。

- 最终清理容器，如果创建时附件参数` --rm`你只需要停止所有正在运行的容器。`docker stop <container-name>`之后它们会自动删除自己。

---

推荐搜索[Dockerhub](https://hub.docker.com/explore/)获取官方构建的基础镜像。例如, in the official [PyTorch image](https://hub.docker.com/r/pytorch/pytorch)。这样再构建自己的Dockerfile时只需要很少的代码。

---

#### 本节作业

* 参考以上方式，通过容器创建之前训练模型的容器

1 提交Dockerfile

2 build镜像，提交镜像构建成功的日志。参考"构建Docker镜像"。

例如：
```
Successfully built 384b59932a42
Successfully tagged train_dl:latest
```

3 启动训练程序，提交训练成功日志。参考"启动所构建镜像的容器实例"。

例如：
```
...
Train Epoch: 1 [49920/60000 (83%)]      Loss: 0.195172
Train Epoch: 1 [50560/60000 (84%)]      Loss: 0.052903
Train Epoch: 1 [51200/60000 (85%)]      Loss: 0.262594
Train Epoch: 1 [51840/60000 (86%)]      Loss: 0.066995
Train Epoch: 1 [52480/60000 (87%)]      Loss: 0.022450
Train Epoch: 1 [53120/60000 (88%)]      Loss: 0.239718
Train Epoch: 1 [53760/60000 (90%)]      Loss: 0.152097
Train Epoch: 1 [54400/60000 (91%)]      Loss: 0.047996
Train Epoch: 1 [55040/60000 (92%)]      Loss: 0.250904
Train Epoch: 1 [55680/60000 (93%)]      Loss: 0.048776
Train Epoch: 1 [56320/60000 (94%)]      Loss: 0.090460
Train Epoch: 1 [56960/60000 (95%)]      Loss: 0.088897
Train Epoch: 1 [57600/60000 (96%)]      Loss: 0.124534
Train Epoch: 1 [58240/60000 (97%)]      Loss: 0.144476
...
```


## 下一步
下一步教程 [3. Docker部署PyTorch推理程序](./inference.md)

#### 参考资料:

- [Dockerfile reference](https://docs.docker.com/engine/reference/builder/)
- [Dockerhub](https://hub.docker.com/)
- [Official node image on Dockerhub](https://hub.docker.com/_/node/)
