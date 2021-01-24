# Lab 6 - 学习使用调度管理系统

## 实验目的

以 [Microsoft Open Platform for AI (OpenPAI)](https://github.com/microsoft/pai) 为例，学习搭建并使用面向深度学习的异构计算集群调度与资源管理系统。

## 实验环境

本实验为分组实验，3~4 位同学一组，实验内容略有差别， 实验流程中将以 Alice, Bob, Carol, Dan 指代（3人组请忽略 Dan），每位同学的环境均为：

* Ubuntu 18.04 LTS
* NVIDIA GPU (已装好驱动)
* Docker Engine
* nvidia-container-runtime
* ssh and sshd

## 实验原理

1. 基于 Kubernetes 的深度学习集群管理系统

    Kubernetes 是一个可移植的、可扩展的开源平台，用于管理容器化的工作负载和服务，可促进声明式配置和自动化。一个 Kubernetes 集群由一组被称作节点的机器组成。这些节点上运行 Kubernetes 所管理的容器化应用，集群具有至少一个主节点和至少一个工作节点。

    主节点管理集群中的工作节点和 Pod，通常运行控制组件，对集群做出全局决策（比如调度），以及检测和响应集群事件，主要包括以下组件：

    * kube-apiserver

        主节点上负责提供 Kubernetes API 服务的组件。

    * etcd

      etcd 是兼具一致性和高可用性的键值数据库，可以作为保存 Kubernetes 所有集群数据的后台数据库。

    * kube-scheduler

      主节点上的组件，该组件监视那些新创建的未指定运行节点的 Pod，并选择节点让 Pod 在上面运行。

    * kube-controller-manager

      在主节点上运行控制器的组件。

    工作节点托管作为应用程序组件的 Pod，维护运行的 Pod 并提供 Kubernetes 运行环境，主要包括以下组件：

    * kubelet

        kubelet 是一个在集群中每个节点上运行的代理，保证容器都运行在 Pod 中。

    * kube-proxy

        kube-proxy 是集群中每个节点上运行的网络代理，维护节点上的网络规则。

    * Container Runtime

        容器运行环境是负责运行容器的软件，例如 Docker.

2. HiveD Scheduler 调度算法

    HiveD Scheduler 是一个适用于多租户 GPU 集群的 Kubernetes Scheduler Extender. 多租户 GPU 群集假定多个租户（团队）在单个物理集群（Physical Cluster）中共享同一 GPU 池，并为每个租户提供一些资源保证。HiveD 将每个租户创建一个虚拟集群（Virtual Cluster），以便每个租户可以像使用私有群集一样使用自己的虚拟集群 VC，同时还可以较低优先级地使用其他租户 VC 的空闲资源。

    HiveD 为 VC 提供资源保证，不仅是资源的数量保证，还提供资源拓扑结构的保证。例如，传统的调度算法可以确保 VC 使用 8 块 GPU，但是它不知道这 8 块 GPU 的拓扑结构，即使在其 VC 仍有 8 个空闲 GPU 的情况下也可能因为这些 GPU 在不同的机器上，无法分配在单个机器上运行的 8 卡训练任务。HiveD 可以为 VC 提供 GPU 拓扑结构的保证，例如保证 VC 可以使用在同一个机器上的 8 块 GPU.

    HiveD 通过 cell 单元来分配资源，一个 cell 单元包含用户自定义的资源数量和硬件的拓扑结构信息。例如用户可以定义一个包含 8 GPU 的节点，并把一个这样的 cell 分配给 VC，这样 HiveD 可以保证该 VC 一定有一个可分配的 8 GPU 机器，不管其它 VC 的资源分配情况怎样。HiveD 支持灵活的 cell 单元定义，来保证细粒度的资源分配。例如，用户可以针对不同的 AI 硬件（例如 NVIDIA V100, AMD MI50, Google Cloud TPU v3）或网络配置（例如 InfiniBand）在多个拓扑层级（例如 PCIe switch, NUMA）定义 cell 单元。VC 可以包含各种层级的 cell 单元，HiveD 可以保证所有 cell 单元的资源。

## 实验内容

### 实验流程图

![](/imgs/Lab6-flow.png "Lab6 flow chat")

### 具体步骤

1. 安装环境依赖

    （以下步骤在 Alice, Bob, Carol, Dan 的机器上执行）

    1. 安装 Docker Engine

        参照 [Docker Engine 文档](https://docs.docker.com/engine/install/ubuntu/) 在 Ubuntu 上安装 Docker Engine.

    2. 安装 nvidia-container-runtime

        参照 [Installation 文档](https://github.com/NVIDIA/nvidia-container-runtime#installation) 在 Ubuntu 上安装 nvidia-container-time
        
        参照[文档](https://github.com/NVIDIA/nvidia-container-runtime#daemon-configuration-file) 修改 Docker daemon 配置文件，将 `default-runtime` 设为 `nvidia`，配置文件修改后需要使用 `sudo systemctl restart docker` 重启 Docker daemon.

    3. 验证安装结果

        * 通过 `sudo docker info` 检查是否有 "Default runtime: nvidia" （默认为 runc）.
        * 通过 `sudo docker run nvidia/cuda:10.0-base nvidia-smi` 运行一个 GPU Docker 看是否能正确看到 GPU 信息。

    4. 新建 Linux 用户

        新建相同的 Linux 用户，例如 username: openpai, password: paiopen, 并将该用户加到 sudo 组里。

        ```sh
        sudo useradd openpai
        sudo usermod -a -G sudo openpai
        ```

2. 部署 OpenPAI

    在部署的集群中：Alice 的机器为 dev-box（管理员用来操作集群，不在集群中），Bob 的机器为 master（在集群中，不跑具体的任务），Carol 和 Dan 的机器为 worker（在集群中，用来跑用户的任务）。

    （以下步骤只在 Alice 的机器上执行）

    1. 准备配置文件

        * `~/master.csv`:

            ```
            hostname-bob,10.0.1.2
            ```

            "hostname-bob" 是在 Bob 的机器上执行 `hostname` 的结果，10.0.1.2 替换为 Bob 的机器的 ip 地址。

        * `~/worker.csv`:

            ```
            hostname-carol,10.0.1.3
            hostname-dan,10.0.1.4
            ```

            "hostname-carol" 是在 Carol 的机器上执行 `hostname` 的结果，10.0.1.3 替换为 Carol 的机器的 ip 地址。Dan 同理。

        * `~/config.yaml`:

            ```yaml
            user: openpai
            password: paiopen
            branch_name: v1.2.0
            docker_image_tag: v1.2.0
            ```

            "user" 和 "password" 是新建的 Linux 用户的 username 和 password.

    2. 部署 OpenPAI

        1. 克隆 OpenPAI 的代码

            ```sh
            git clone -b v1.2.0 https://github.com/microsoft/pai.git
            cd pai/contrib/kubespray
            ```

        2. 部署 Kubernetes

            ```sh
            bash quick-start-kubespray.sh -m ~/master.csv -w ~/worker.csv -c ~/config.yaml
            ```

        3. 启动 OpenPAI 服务

            ```sh
            bash quick-start-service.sh -m ~/master.csv -w ~/worker.csv -c ~/config.yaml
            ```

        如果部署成功，会看到如下信息：
        ```
        Kubernetes cluster config :     ~/pai-deploy/kube/config
        OpenPAI cluster config    :     ~/pai-deploy/cluster-cfg
        OpenPAI cluster ID        :     pai
        Default username          :     admin
        Default password          :     admin-password

        You can go to http://<your-master-ip>, then use the default username and password to log in.
        ```

        在浏览器中访问 http://bob-ip，使用 admin 和 admin-password 登陆。

    3. 运行 dev-box Docker 管理集群

        运行 dev-box Docker 容器：
        ```sh
        sudo docker run -itd \
            -v /var/run/docker.sock:/var/run/docker.sock \
            -v ${HOME}/pai-deploy/cluster-cfg:/cluster-configuration  \
            -v ${HOME}/pai-deploy/kube:/root/.kube \
            --privileged=true \
            --name=dev-box \
            openpai/dev-box:v1.2.0
        ```

        执行 dev-box Docker 容器：
        ```sh
        sudo docker exec -it dev-box bash
        ```

        列出集群中的节点：
        ```sh
        kubectl get nodes
        ```

        使用 `paictl` 管理 OpenPAI 服务：
        ```sh
        cd /pai
        python paictl.py config get-id
        ```

    > 注：详细的部署说明请参考 [Installation Guide](https://openpai.readthedocs.io/en/latest/manual/cluster-admin/installation-guide.html#installation-from-scratch)，部署过程中遇到的问题可以参考 [troubleshooting](https://openpai.readthedocs.io/en/latest/manual/cluster-admin/installation-faqs-and-troubleshooting.html#troubleshooting) 或在 [GitHub issue](https://github.com/microsoft/pai/issues) 上提问。

3. 使用 OpenPAI

    1. 新建 OpenPAI 用户

        （Bob 执行）

        Bob 访问 http://bob-ip, 在 Administration -> User Management 页面中，给 Alice, Carol, Dan 分别新建账号。

    2. 提交集群任务

        （Alice, Bob, Carol, Dan 都执行）

        在浏览器中访问 http://bob-ip, 在 Submit Job 页面提交 Single Job。观察集群中任务的等待和执行情况。

4. 更改调度器配置并使用不同调度策略

    1. 更改调度器配置

        （Alice 执行）

        将两个 GPU 机器配置成两个不同的 VC，在 dev-box Docker 容器中更改 `/cluster-configuration/service-configuration.yaml` 文件中的 `hivedscheduler`:

        ```yaml
        hivedscheduler:
          config: |
            physicalCluster:
              skuTypes:
                GPU-C:
                  gpu: 1
                  cpu: 2
                  memory: 4096Mi
                GPU-D:
                  gpu: 1
                  cpu: 2
                  memory: 4096Mi
              cellTypes:
                GPU-C-NODE:
                  childCellType: GPU-C
                  childCellNumber: 1
                  isNodeLevel: true
                GPU-C-NODE-POOL:
                  childCellType: GPU-C-NODE
                  childCellNumber: 1
                GPU-D-NODE:
                  childCellType: GPU-D
                  childCellNumber: 1
                  isNodeLevel: true
                GPU-D-NODE-POOL:
                  childCellType: GPU-D-NODE
                  childCellNumber: 1
              physicalCells:
              - cellType: GPU-C-NODE-POOL
                cellChildren:
                - cellAddress: hostname-carol #TODO change to Carol's
              - cellType: GPU-D-NODE-POOL
                cellChildren:
                - cellAddress: hostname-dan #TODO change to Dan's
            virtualClusters:
              default:
                virtualCells:
                - cellType: GPU-C-NODE-POOL.GPU-C-NODE
                  cellNumber: 1
              vc1:
                virtualCells:
                - cellType: GPU-D-NODE-POOL.GPU-D-NODE
                  cellNumber: 1
        ```

        然后使用 `paictl` 更新配置文件并重启相应的服务（提示输入的 cluster-id 为 "pai"）：

        ```sh
        python paictl.py service stop -n rest-server hivedscheduler
        python paictl.py config push -p /cluster-configuration -m service
        python paictl.py service start -n rest-server hivedscheduler
        ```

    2. VC 安全 (VC Safety)

        （Alice, Bob, Carol, Dan 都执行，可同时进行）

        同时向 `vc1` 提交任务（任务配置文件可参考 `job-config-0.yaml`），观察任务的运行情况：提交的任务会在哪个机器上运行，当有多个任务在等待并且集群中的 `default` VC 空闲时任务会被怎样调度？

    3. 优先级和抢占 (Priority and Preemption)

        （Alice, Bob, Carol, Dan 按顺序依次实验，实验时确保集群中没有其它未结束的任务）

        先向 `vc1` 提交一个优先级 `jobPriorityClass` 为 `test` 的任务（任务配置文件可参考 `job-config-1.yaml`），在其运行时再向 `vc1` 提交一个优先级为 `prod` 的任务（任务配置文件可参考 `job-config-2.yaml`），观察任务的运行情况：后提交的任务是否在先提交的任务运行完成之后运行，什么时候两个任务都运行结束？

    4. 低优先级任务 (Opportunistic Job)

        （Alice, Bob, Carol, Dan 按顺序依次实验，实验时确保集群中没有其它未结束的任务）

        先向 `vc1` 提交一个优先级 `jobPriorityClass` 为 `prod` 的任务（任务配置文件可参考 `job-config-3.yaml`），在其运行时再向 `vc1` 提交一个优先级为 `oppo`（最低优先级）的任务（任务配置文件可参考 `job-config-4.yaml`），观察任务的运行情况：后提交的任务什么时候开始运行，是否会等高优先级的任务运行完？如果在后提交的任务运行时再向 `default` VC 提交优先级为 `test` 的任务会被怎样调度？

    5. 更改调度器配置

        （Alice 执行）

        将两个 GPU 机器配置在相同 VC 里，在 dev-box Docker 容器中更改 `/cluster-configuration/service-configuration.yaml` 文件中的 `hivedscheduler`:

        ```yaml
        hivedscheduler:
          config: |
            physicalCluster:
              skuTypes:
                GPU:
                  gpu: 1
                  cpu: 2
                  memory: 4096Mi
              cellTypes:
                GPU-NODE:
                  childCellType: GPU
                  childCellNumber: 1
                  isNodeLevel: true
                GPU-NODE-POOL:
                  childCellType: GPU-NODE
                  childCellNumber: 2
              physicalCells:
              - cellType: GPU-NODE-POOL
                cellChildren:
                - cellAddress: hostname-carol #TODO change to Carol's
                - cellAddress: hostname-dan #TODO change to Dan's
            virtualClusters:
              default:
                virtualCells:
                - cellType: GPU-NODE-POOL.GPU-NODE
                  cellNumber: 2
        ```

        然后使用 `paictl` 更新配置文件并重启相应的服务（提示输入的 cluster-id 为 "pai"）：

        ```sh
        python paictl.py service stop -n rest-server hivedscheduler
        python paictl.py config push -p /cluster-configuration -m service
        python paictl.py service start -n rest-server hivedscheduler
        ```

    6. 群调度 (Gang Scheduling)

        （Alice, Bob, Carol, Dan 按顺序依次实验，实验时确保集群中没有其它未结束的任务）

        先向 `default` VC 提交一个任务占用一台机器（任务配置文件可参考 `job-config-5.yaml`），在其运行时再向 `default` VC 提交一个有 2 个子任务需要两台机器的任务（任务配置文件可参考 `job-config-6.yaml`），观察任务的运行情况：后提交的任务什么时候开始运行，2 个子任务是否会先后运行？

    7. 弹性调度 (Incremental Scheduling)

        （Alice, Bob, Carol, Dan 按顺序依次实验，实验时确保集群中没有其它未结束的任务）

        先向 `default` VC 提交一个任务占用一台机器，在其运行时再向 `default` VC 提交一个有 2 个子任务需要两台机器的任务（任务配置文件可参考 `job-config-7.yaml`），观察任务的运行情况：后提交的任务什么时候开始运行，2 个子任务是否会先后运行？能否在当前只有 2 GPU 的集群中提交一个需要用超过配额（例如用 4 GPU）的任务？

## 实验报告

### 实验环境

（Alice/Bob/Carol/Dan 替换为组员姓名）
|||||||
|-------|-|-------|-----|------|------|
| Users | | &nbsp; &nbsp; &nbsp; &nbsp; Alice &nbsp; &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; Bob &nbsp; &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; Carol &nbsp; &nbsp; &nbsp; &nbsp; | &nbsp; &nbsp; &nbsp; &nbsp; Dan &nbsp; &nbsp; &nbsp; &nbsp; |
| 硬件环境 | CPU（vCPU数目）||||||
|| GPU(型号，数目) |||||
|| IP |||||
|| HostName |||||
| 软件环境 | OS版本 |||||
|| Docker Engine版本 |||||
|| CUDA版本 |||||
|| OpenPAI版本 |||||
|||||||

### 实验结果

 1. 部署 OpenPAI

     简述部署中遇到的问题以及相应的解决方案。
     <br/>

     <br/>

     <br/>

     <br/>

     <br/>

 2. 使用不同调度策略
     ||||
     | --- | --- | --- |
     | 实验名称 | 实验现象（任务运行情况） | 支持文件（任务配置文件, UI 截图等） |
     | VC 安全 (VC Safety) | 提交的任务会在哪个机器上运行，当有多个任务在等待并且集群中的 `default` VC 空闲时任务会被怎样调度？其它观察到的现象 | |
     | 优先级和抢占 (Priority and Preemption) | 后提交的任务是否在先提交的任务运行完成之后运行，什么时候两个任务都运行结束？其它观察到的现象 | |
     | 低优先级任务 (Opportunistic Job) | 后提交的任务什么时候开始运行，是否会等高优先级的任务运行完？如果在后提交的任务运行时再向 `default` VC 提交优先级为 `test` 的任务会被怎样调度？其它观察到的现象 | |
     | 群调度 (Gang Scheduling) | 后提交的任务什么时候开始运行，2 个子任务是否会先后运行？其它观察到的现象 | |
     | 弹性调度 (Incremental Scheduling) | 后提交的任务什么时候开始运行，2 个子任务是否会先后运行？能否在当前只有 2 GPU 的集群中提交一个需要用超过配额（例如用 4 GPU）的任务？其它观察到的现象 | |
     ||||

## 参考代码

代码位置：`Lab6/config`

## 参考资料

* Open Platform for AI (OpenPAI): https://github.com/microsoft/pai
* OpenPAI Handbook: https://openpai.readthedocs.io/en/latest/
* HiveD Scheduler: https://github.com/microsoft/hivedscheduler
* Kubernetes Documentation: https://kubernetes.io/docs/home/