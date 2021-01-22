# MIT License

# Copyright (c) Microsoft Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

# 继承自哪个基础镜像
FROM ubuntu:18.04

# 创建镜像中的文件夹，用于存储新的代码或文件
RUN mkdir -p /src/app

# WORKDIR指令设置Dockerfile中的任何RUN，CMD，ENTRPOINT，COPY和ADD指令的工作目录
WORKDIR /src/app

# 拷贝本地文件到Docker镜像中相应目录
COPY pytorch_mnist_basic.py /src/app

# 需要安装的依赖 

RUN apt-get update && apt-get install wget bzip2 -y
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda
ENV PATH /opt/conda/bin:$PATH

RUN conda install pytorch torchvision cpuonly -c pytorch

# 容器启动命令
CMD [ "python", "pytorch_mnist_basic.py" ]
