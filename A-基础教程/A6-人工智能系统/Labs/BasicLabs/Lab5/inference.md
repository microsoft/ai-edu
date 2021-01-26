# Docker部署PyTorch推理程序
## 创建TorchServe镜像

### 准备TorchServe源码:
```bash
# $BRANCH_NAME可以指定需要使用的版本
$ rm -rf serve
$ git clone https://github.com/pytorch/serve.git
$ cd serve
$ git checkout $BRANCH_NAME
$ cd ..
```

#### 创建基于CPU镜像:
```bash
$ docker build --file Dockerfile.infer.cpu -t torchserve:0.1-cpu .
```

如果成功，显示类似日志：
```
...
Successfully built e36f1a01e514
Successfully tagged torchserve:0.1-cpu
```

#### 创建基于GPU镜像:
```bash
$ docker build --file Dockerfile.infer.gpu -t torchserve:0.1-gpu .
```

如果成功，显示类似日志：
```
Successfully built 456de47eb88e
Successfully tagged torchserve:0.1-gpu
```
## 使用TorchServe镜像启动一个容器

以下的实例会启动容器并打开8080/81端口，并暴露给主机

#### 启动CPU容器

如果想用特定的版本，可以传递特定的标签确定使用(ex: 0.1-cpu):

```bash
$ docker run --rm -it -p 8080:8080 -p 8081:8081 torchserve:0.1-cpu
```

启动结果，进入控制台：

```
root@d73100253567:/#
```

对官方最新的版本，你可以使用`latest`标签:

```
$ docker run --rm -it -p 8080:8080 -p 8081:8081 pytorch/torchserve:latest
```

#### 启动GPU容器

如果想用特定的版本，可以传递特定的标签确定使用(ex: 0.1-cuda10):

```bash
$ docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 pytorch/torchserve:0.1-cuda10.1-cudnn7-runtime
```

对最新的官方版本，你可以使用`gpu-latest`标签:

```bash
$ docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 pytorch/torchserve:latest-gpu
```

#### 容器内部访问TorchServe APIs

TorchServe的推理和管理APIs可以通过主机的8080和8081端口访问。例如：

```bash
# 访问服务
$ curl http://localhost:8080/ping
```

返回正常状态：
```
{
  "status": "Healthy"
}

```

#### 在主机检查正在运行的容器

```bash
$ docker ps
```

返回结果：
```
CONTAINER ID        IMAGE                                                         COMMAND                  CREATED              STATUS              PORTS                              NAMES
7d3f0e9be89a        torchserve:0.1-cpu                                            "/usr/local/bin/dock…"   About a minute ago   Up About a minute   0.0.0.0:8080-8081->8080-8081/tcp   sad_engelbart
```

#### 进入容器

执行进入容器命令：

```
$ docker exec -it <containerid> /bin/bash
```

返回结果：

```
# 进入容器控制台
root@7d3f0e9be89a:/home/model-server#
```

#### 停止TorchServe容器

```bash
$ docker container stop <containerid>
```

Container ID可以通过`docker ps`命令查询。

#### 检查容器的端口映射

```bash
$ docker port <containerid>
```

返回结果：
```
8081/tcp -> 0.0.0.0:8081
8080/tcp -> 0.0.0.0:8080
```
#### 重要提示：

If you are hosting web-server inside your container then explicitly specify the ip/host as 0.0.0.0 for your web-server
For details, refer : https://docs.docker.com/v17.09/engine/userguide/networking/default_network/binding/#related-information

#### 本节作业

* 参考以上方式，通过容器创建之前训练模型的推理服务

参考实例：
##### 保存一个模型

使用TorchServe推理，第一步需要把模型使用model archiver归档为MAR文件，

1. 进入容器

```
$ docker exec -it <containerid> /bin/bash
```

1. 创建或进入保存模型的文件夹.

    ```bash
    $ cd /home/model-server/model-store
    $ mkdir model-store
    ```

2. 下载一个模型.

    ```bash
    $ apt-get update  
    $ apt-get install wget
    $ wget https://download.pytorch.org/models/densenet161-8d451a50.pth
    ```

    返回结果：
    ```
    20xx-0x-0x 05:13:05 (84.6 MB/s) - 'densenet161-8d451a50.pth' saved [115730790/115730790]
    ```

3. 使用model archiver进行模型归档. The `extra-files` param uses fa file from the `TorchServe` repo, so update the path if necessary.
    安装：
    ```
    $ cd /serve/model-archiver
    $ pip install .
    ```

    ```bash
    $ torch-model-archiver --model-name densenet161 --version 1.0 --model-file /serve/examples/image_classifier/densenet_161/model.py --serialized-file /home/model-server/model-store/densenet161-8d451a50.pth --export-path /home/model-server/model-store --extra-files /serve/examples/image_classifier/index_to_name.json --handler image_classifier
    ```

    成功后可以看到mar文件：
    ```
    # 执行ls
    densenet161-8d451a50.pth  densenet161.mar
    ```

##### 启动TorchServe进行推理模型

After you archive and store the model, use the `torchserve` command to serve the model.

关闭之前的如果已经启动的TorchServe
```
$ torchserve --stop
```

如果停止成功：
```
TorchServe has stopped.
```

```bash
$ torchserve --start --ncs --model-store model-store --models densenet161.mar
```

如果启动成功，会观察到类似以下日志： 
```
e89a,timestamp:1591593790
2020-06-08 05:23:10,347 [INFO ] W-9009-densenet161_1.0 org.pytorch.serve.wlm.WorkerThread - Backend response time: 6595
2020-06-08 05:23:10,348 [DEBUG] W-9009-densenet161_1.0 org.pytorch.serve.wlm.WorkerThread - W-9009-densenet161_1.0 State change WORKER_STARTED -> WORKER_MODEL_LOADED
2020-06-08 05:23:10,348 [INFO ] W-9009-densenet161_1.0 TS_METRICS - W-9009-densenet161_1.0.ms:6816|#Level:Host|#hostname:7d3f0e9be89a,timestamp:1591593790
2020-06-08 05:23:10,358 [INFO ] W-9010-densenet161_1.0 org.pytorch.serve.wlm.WorkerThread - Backend response time: 6606
2020-06-08 05:23:10,358 [DEBUG] W-9010-densenet161_1.0 org.pytorch.serve.wlm.WorkerThread - W-9010-densenet161_1.0 State change WORKER_STARTED -> WORKER_MODEL_LOADED
2020-06-08 05:23:10,358 [INFO ] W-9010-densenet161_1.0 TS_METRICS - W-9010-densenet161_1.0.ms:6826|#Level:Host|#hostname:7d3f0e9be89a,timestamp:1591593790
2020-06-08 05:23:10,362 [INFO ] W-9006-densenet161_1.0 org.pytorch.serve.wlm.WorkerThread - Backend response time: 6610
2020-06-08 05:23:10,362 [DEBUG] W-9006-densenet161_1.0 org.pytorch.serve.wlm.WorkerThread - W-9006-densenet161_1.0 State change WORKER_STARTED -> WORKER_MODEL_LOADED
2020-06-08 05:23:10,362 [INFO ] W-9006-densenet161_1.0 TS_METRICS - W-9006-densenet161_1.0.ms:6832|#Level:Host|#hostname:7d3f0e9be89a,timestamp:1591593790
```

##### 使用模型进行推理

为了测试模型服务，发送请求到服务器的`predictions` API.

参考以下步骤执行:

* 开启新的终端窗口 
* 使用`curl`命令下载一个实例[cute pictures of a kitten](https://www.google.com/search?q=cute+kitten&tbm=isch&hl=en&cr=&safe=images)
  并且通过 `-o` f标识重命名其为`kitten.jpg`.
* 使用`curl`发送kitten图像，`POST`到TorchServe `predict`终端入口.

下面的代码完成了所有的三个步骤

在主机客户端执行以下命令：

```bash
$ curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
$ curl -X POST http://127.0.0.1:8080/predictions/densenet161 -T kitten.jpg
```

预测的终端可以返回JSON格式的应答. 例如下面的例子:

```json
[
  {
    "tiger_cat": 0.46933549642562866
  },
  {
    "tabby": 0.4633878469467163
  },
  {
    "Egyptian_cat": 0.06456148624420166
  },
  {
    "lynx": 0.0012828214094042778
  },
  {
    "plastic_bag": 0.00023323034110944718
  }
]
```

请同学参考以上步骤，进行自己的模型的服务部署，并返回推理请求的结果截图。

