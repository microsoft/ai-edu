Copyright © Microsoft Corporation. All rights reserved.
  适用于[License](https://github.com/Microsoft/ai-edu/blob/master/LICENSE.md)版权许可


# AI对联生成
-------

# 场景描述

## 对联的由来及特点

对联，也称“楹联”、“对子”，是一种由字数相同的两句话组成的对仗工整、韵律协调、语义完整的文学形式。它发源于我国古诗的对偶句，始创于五代时期，盛于明清，至今已有一千多年的历史了。对联的形式工整、平仄协调的特点，是一字一音、音形义统一的汉字特色的体现，所以，对联是汉语语言特有的文学形式，是中华民族的文化瑰宝，是我国的重要文化遗产。

在我国民间，对联有着广泛的应用。比如，过年时家门上贴春联，商店开业时门上挂对联，以及娱乐时的对对联游戏。

对联的长度不定，短的可以只有一两个字；长的则可达几百个字。

## 对联的自动生成

### 使用统计机器学习

在基于统计的机器翻译中，规则是由机器自动从大规模的语料中学习得到的，而非由人主动提供完整的规则。

微软亚洲研究院周明老师团队早在十几年前就已经使用基于短语的统计机器学习的方法，实现了电脑自动对联系统，效果非常好，很好的展现了中国经典文化的魅力，收获了非常多的赞誉。在线体验地址是 [这里](http://duilian.msra.cn)。

### 使用深度学习

近年来，深度神经网络学习的发展为机器翻译提供了新的思路。通常情况下，神经机器翻译使用编码器-解码器框架。编码阶段将整个源序列编码成一个（或一组）向量，解码阶段通过最大化预测序列概率，从中解码出整个目标序列，完成翻译过程。

![编码解码过程](./images/encode-decode.png "编码解码过程")

编码器、解码器通常使用 RNN、LSTM 来实现，也有的用 CNN 来实现，达到了比较好的性能和结果。

在本次案例中，我们使用深度学习的方法，实现一个对联自动生成微信小程序 —— ***联景联情***

可以在微信中搜索 “联景联情” 找到并使用该小程序，或者扫描如下二维码获取。

![联景联情](./src/imgs/qrcode.jpg "扫码使用小程序")

# 案例概要

## 案例描述

用户在小程序中轻松上传一张图像，程序提取图像信息，自动生成3组5-7字的备选上下联。用户可选择一组心仪的对联，与图像合成你的专属对联。

该程序是由学习微软亚洲研究院首席研发经理邹欣老师《软件工程实践》课程的几位同学（他们均在微软亚洲研究院实习）组队完成的，在完成过程中充分利用课程所学软件工程知识，结合NLP知识和软件开发技能，完成了一个端到端的应用服务。这里可以看到他们的开发感想与总结。

## 程序结构

本案例的基本程序结构如下图所示：

![程序结构图](./images/codeflow.PNG)

后续将会对每个部分进行详细说明。

## 涉及知识

* 使用微软认知服务（Cognitive Service）中计算机视觉（computer vision）服务
* NLP 相关知识
   * Sequence2Sequence 模型
   * 相关算法： RNN，LSTM，GRU，Transformer
* 模型库的使用
   * Tensor2Tensor
   * Fairseq

## 案例价值

此案例特色显明，生动有趣，可以激发学生们对深度学习的兴趣。在技术层面，此案例使学生对深度学习的时序模型有直观的了解。该案例面向对象广泛，扩展性强。对初学者，可重复案例的过程；对于进阶者，不论在模型选择上，还是在模型推理上，都可以有更多的扩展，可提高学生们的探索研究能力。


# 案例详解

## 搭建环境

### 证书许可

由于微信小程序使用https协议访问服务器，需要申请SSL证书。本案例申请了微软的SSL证书。

### 操作系统

本案例运行在Azure虚拟机上，虚拟机的系统为Ubuntu 16.04

### 编程语言

本案例的语言使用 Python3.x ，并需要安装一些 python packages。

1. 训练所需 python packages 在文件 [train_requirements.txt](./src/training/train_requirerments.txt) 中。
2. 服务所需 python packages 在文件 [conda_requirements.txt](./src/service/conda_requirements.txt) 中。

### 框架和模型库

本案例使用 tensorflow 的 tensor2tensor 模型库，具体版本如下：
- tensorflow 1.4.0
- tensor2tensor 1.2.9

## 模型训练

### 模型选择

想要完成一个自动生成对联的小程序，想法十分美好，但想要达到这个目标，光拍拍脑袋想想是不够的，需要训练出一个能完成对联生成的自然语言理解模型。于是乎，就有两个选择：

1. 自己写一套完成对联生成工作的深度学习模型。这个工作量相当之大，可能需要一个NLP专业团队来进行开发，调优。
2. 应用已有的深度学习模型，直接应用。这个选择比较符合客观需要。我们找到了两个工具包：

    + Tensor2Tensor 工具包：Tensor2Tensor（以下简称T2T）是由 Google Brain 团队使用和维护的开源深度学习模型库，支持多种数据集和模型。T2T 在 github 上有完整的介绍和用法，可以访问[这里](https://github.com/tensorflow/tensor2tensor)了解详细信息。

    + Fairseq 工具包：[Fairseq](https://github.com/pytorch/fairseq) 是 Facebook 推出的一个序列建模工具包，这个工具包允许研究和开发人员自定义训练翻译、摘要、语言模型等文本生成任务。这里是它的 PyTorch 实现。

    本案例中，我们使用 T2T 工具包进行模型训练。


### 数据收集

有了模型，还需要数据。巧妇难为无米之炊，没有数据，什么都是浮云。数据从哪里来呢？GitHub 上有很多开源贡献者收集和整理了对联数据，可以进行下载使用。

本案例从下面几个渠道获取对联数据：
1. Github网站上的开源对联数据： https://github.com/wb14123/couplet-dataset/releases
2. Github网站上的开源古诗数据： https://github.com/chinese-poetry/chinese-poetry
3. 微软亚洲研究院提供的10万条对联数据（非公开数据）。

### 数据预处理

#### 生成源数据文件

网上提供的对联数据形式各异，需要整理成我们需要的格式。我们创建两个文本文件，命名为 train.txt.up 和 train.txt.down，存放上联和下联数据。每个上联/下联为一行，用换行符 ‘\n’ 分隔。

#### 生成词表文件

接下来我们要统计上下联中出现多少不同的字，用于后续的模型推理。
1. 将上下联数据每个字以“空格”分隔，合并成一个文件。

    a. 分隔数据的python代码 (split_data.py)：
    ```
    import sys

    filename = sys.argv[1]
    with open(filename, 'r', encoding='utf-8') as infile:
        with open(filename + '.clean', 'w', encoding='utf-8') as outfile:
            lines = infile.readlines()
            for line in lines:
                out = ""
                for i in line.strip():
                    out += i + (' ')
                out = out[:-1]
                out += '\n'
                outfile.write(out)
    ```
    b. 执行如下命令完成文件分隔
    ```
    python split_data.py train.txt.up
    python split_data.py train.txt.down
    ```
    分隔后生成两个文件：train.txt.up.clean 和 train.txt.down.clean

    c. 合并文件为 merge.txt
    ```
    cat train.txt.up train.txt.down > merge.txt
    ```
2. 统计文件中出现的不同字和每个字的出现次数。
    ```
    subword-nmt get-vocab –input merge.txt –output merge.txt.vocab
    ```
3. 去掉出现次数，只保留字
    ```
    cat merge.txt.vocab | awk ‘{print $1}’ > merge.txt.vocab.clean
    ```
4. 将 merge.txt.vocab.clean 的前三行填充如下内容，并将字表字数加3：
    ```
    <pad>
    <EOS>
    <UNK>
    ```

5. 生成测试集。

    取训练集中前 100 个数据作为测试集。（在实际训练过程中，没有用到测试集）
    ```
    head -n 100 train.txt.up > dev.txt.up
    head -n 100 train.txt.down > dev.txt.down
    ```

#### 下载指定版本模型库

```
git clone https://github.com/tensorflow/tensor2tensor.git
git checkout v1.2.9
```

#### 编写问题定义文件

本案例在 merge_vocab.py 文件中编写了下联生成模型的问题定义。

文件中定义了如下参数：
1. SRC_TRAIN_DATA 为训练集上联数据文件
2. TGT_TRAIN_DATA 为训练集下联数据文件
3. SRC_DEV_DATA 为测试集上联数据文件
4. TGT_DEV_DATA 为测试集下联数据文件
5. MERGE_VOCAB 为最终字表文件
6. VOCAB_SIZE为字表文件中字的个数

并注册了问题类 ```TranslateUp2down``` ，用于指出如何进行上下联翻译。其中 ```generator``` 函数用于处理词表、编码、创建完成时序任务的生成器的工作。


#### 生成训练数据

在本案例中，若要使用 T2T 工具包进行训练，需要把数据转换成T2T认可的二进制文件形式。T2T 工具包提供了生成训练数据的命令：`t2t_datagen` 命令，本案例中使用的具体命令和参数如下：
```
python tensor2tensor/bin/t2t-datagen \
  --t2t_usr_dir=${DATA_DIR} \
  --data_dir=${DATA_DIR} \
  --problem=${PROBLEM}
```
其中，

*t2t_usr_dir*：指定了一个目录，该目录中包涵 \_\_init\_\_.py 文件，并可以导入处理对联问题的 python 模块。本案例中创建一个 data 目录，并将其均放入此目录。在该目录中，编写 merge_vocab.py 文件，注册对联问题。并添加一个 \_\_init\_\_.py文件，将 merge_vocab.py 作为模块导入。

*data_dir*：数据目录。存放生成训练数据所需的所有源数据资源，以及生成的训练数据文件。

*problem*：定义问题名称，本案例中问题名称为 translate_up2down
当命令执行完毕，将会在 data 目录下生成两个文件：

    translate_up2down-train-00000-of-00001
    translate_up2down-dev-00000-of-00001

这便是我们需要的训练数据文件。

#### 训练模型

有了处理好的数据，我们就可以进行训练了。训练过程依然调用t2t模型训练命令：`t2t_trainer`。具体命令如下：
```
python tensor2tensor/bin/t2t-trainer \
--t2t_usr_dir=${USR_DIR} \
--data_dir=${DATA_DIR} \
--problems=${PROBLEM} \
--model=${MODEL} \
--hparams_set=${HPARAMS_SET} \
--output_dir=${TRAIN_DIR} \
--keep_checkpoint_max=1000 \
--worker_gpu=1 \
--train_steps=200000 \
--save_checkpoints_secs=1800 \
--schedule=train \
--worker_gpu_memory_fraction=0.95 \
--hparams="batch_size=1024" 2>&1 | tee -a ${LOG_DIR}/train_default.log
```

各项参数的作用和取值分别如下：

1) *t2t_usr_dir*：如前一小节所述，指定了处理对联问题的模块所在的目录。

2) *data_dir*：训练数据目录

3) *problems*：问题名称，即translate_up2down

4) *model*：训练所使用的 NLP 算法模型，本案例中使用 transformer 模型

5) *hparams_set*：transformer 模型下，具体使用的模型。transformer 的各种模型定义在 tensor2tensor/models/transformer.py 文件夹内。本案例使用 transformer_small 模型。

6) *output_dir*：保存训练结果

7) *keep_checkpoint_max*：保存 checkpoint 文件的最大数目

8) *worker_gpu*：是否使用 GPU，以及使用多少 GPU 资源

9) *train_steps*：总训练次数

10) *save_checkpoints_secs*：保存 checkpoint 的时间间隔

11) *schedule*：将要执行的 `tf.contrib.learn.Expeiment` 方法，比如：train, train_and_evaluate, continuous_train_and_eval,train_eval_and_decode, run_std_server

12) *worker_gpu_memory_fraction*：分配的 GPU 显存空间

13) *hparams*：定义 batch_size 参数。

好啦，我们输入完命令，点击回车，训练终于 跑起来啦！如果你在拥有一块 K80 显卡的机器上运行，只需5个小时就可以完成训练。如果你只有 CPU ，那么你只能多等几天啦。
我们将训练过程运行在 Microsoft OpenPAI 分布式资源调度平台上，使用一块 K80 进行训练。4小时24分钟后，训练完成，得到如下模型文件：
   - checkpoint
   - model.ckpt-200000.data-00000-of-00003
   - model.ckpt-200000.data-00001-of-00003
   - model.ckpt-200000.data-00002-of-00003
   - model.ckpt-200000.index
   - model.ckpt-200000.meta

我们将使用该模型文件进行模型推理。


## 模型推理

本案例中，自己编写了一个模型推理类，命名为 up2down_class.py，并将其放在 tensor2tensor/bin 目录下。

该类主要实现了一个函数 `get_next`。该函数以上联语句为输入参数，通过调用 T2T 的 decoding 方法（在 tensor2tensor/bin/utils/ 目录下，decoding.py 文件），生成下联语句。

为了将上联语句作为输入，传输给 decoding 类，我们修改 decoding.py 文件的内容：

1. 在 `decode_from_file` 函数中，加入参数 `input_sentence`，传入上联语句。
2. 在 `_get_sorted_inputs` 函数中，加入参数 `input_sentence`。并将该参数赋值给inputs数组，取代原来从 `decode_hp.delimiter` 参数中获取输入信息的方法。
3. 注释 `decode_from_file` 函数的最后五行，并添加语句 `return decodes[sorted_keys[0]]`。将结果写入 decode 文件的做法更改为直接传回结果数据。


## 应用程序编写

应用程序的编写分为两部分：（1）微信小程序部分——我们称之为前端应用。（2）应用服务部分——我们称之为后端服务。

下面我们先来看一下后端服务做了什么。

### 后端服务

#### 实体提取

当用户通过小程序上传图片或照片时，程序需要从图片中提取出能够描述图片的信息。本案例编写了utils.py文件。其中 do_upload_image 函数完成从上传的图片中提取实体的工作。具体过程如下：

1. 将传入的图片，去掉网络报头，提取图片文件的二进制信息，并存入 raw_images 目录，供后续合成图片时使用。
2. 调用微软认知服务（Cognitive Service）中的计算机视觉服务（Computer Vision），完成图片中实体的提取。本案例编写 cognitive_service.py 文件，编写 `call_cv_api` 函数来完成调用过程。
为了使用微软认知服务，用户需要在微软认知服务的网站上申请计算机视觉服务。可以申请30天试用版，也可以创建 Azure 账号，申请免费的服务。
3. 调用结束，返回的结果包含了提取出的实体信息。一张图片可以提取多个实体，组成实体数组。


#### 上联匹配

提取完实体信息，我们要找出与实体相匹配的上联数据。find_shanglian 函数实现了该需求，具体函数实现在 word_matching.py 中，感兴趣的同学可以查看源代码。

##### 使用数据

在上联匹配中，我们需要用到如下几个数据文件，它们的描述如下表：

---
文件名 | 描述 | 格式 |
:----:|------|------
**train.txt.up** | 存储5-7字上联数据 | 文本格式，每行一个上联，‘\n’为分隔符。（如果句中有逗号，按两句分开处理）
**en2cn_dict.txt** | 实体字词中英文翻译，cache 文件 | 以词典形式存储，初始为空。key 为英文实体字词，value为中文翻译结果
**synonyms_words_dict.txt** | 同义词表，保存实体中文字词的三个同义词，cache 文件 | 以词典形式存储，初始为空。key 为中文实体词语，value 为同义词
**dict_1.txt** | 单个字在上联中出现的位置 | 以词典形式存储。key 为单个字，value 为数组，表示在train.txt.up 中的第几条中出现
**dict_2.txt** | 实体词语在上联中出现的位置 | 以词典形式存储。key 为一个词语，value 为数组，表示在train.txt.up 中的第几条中出现

---

其中，dict_1.txt 和dict_2.txt 文件需要根据上联数据文件 train.txt.up 进行处理。
- dict_1：统计上联数据出现的每个字，作为 key 。并找到 train.txt.up 中含有该字的上联 ID，将该 ID 作为 value 数组的元素，生成词表。
- dict_2：将上联两两相连的字组成词语（有可能两个相连字并不能称为词语，但依然组合在一起），作为key。并找到 train.txt.up 中含有该词语的上联 ID ，将该 ID 作为 value 数组的元素，生成词表。


##### 传入参数

函数 find_shanglian 传入至少5个参数：

（1） 实体信息

（2） 上联数据

（3） dict_1 文件，初始化为空，保存单个字在上联中出现的位置

（4） dict_2 文件，初始化为空，保存一个词语在上联中出现的位置

（5） 返回上联结果的数目

##### 实体标签翻译

从微软认知服务得到的实体标签（Tag）都是英文的，需要先翻译成中文。程序调用有道 API 完成中英翻译（在 Translate 函数中实现），并将{英文标签：中文翻译}保存在词典文件 en2cn_dict.txt 中。每次先在该文件中查找有否翻译完成的实体，如没有，再调用有道API。

##### 查找同义词

为了更多找到相关对联，我们还需要对实体词语进行同义词扩展。这用到名为 **synonyms** 的 python 包。该包的提供 nearby 方法，寻找并返回输入词语的同义词，以及他们的得分。本案例针对对每个实体标签，找到并保留至多三个得分大于 **0.7** 的同义词，并将结果在文件 synonyms_words_dict.txt 文件中缓存，方便下次查找。

##### 随机筛选词语

对每个实体，都找到至多三个同义词，并保存在同一个数组中。程序可以从当前数组包含的词语中，随机筛选75%的词汇，生成最终实体词汇列表，用于后续操作。剩下的词汇，保存在备用列表中。

##### 遍历词表

将实体词汇列表中的实体在词表文件 dict_1.txt 和 dict_2.txt 中进行遍历，找到含有该词汇的上联ID。具体步骤如下：

1. 将实体Tag拆分成单个字，在 dict_1 中遍历，寻找包含该词语的上联。

2. 将实体Tag过滤掉超过两个字的Tag，剩下的在 dict_2 中遍历，寻找包含该词语的上联。

3. 合并两次得到的上联 ID 结果，去重，并保存在 results 字典中，key = 上联 ID，value = 出现次数，作为得分。出现次数越多，得分越高。

4. 将结果按出现次序从高到低排序。

5. 返回指定数目的上联 ID，存在数组中。指定数目=要求返回上联数目+10。如果程序要求返回5个结果，则在这里返回15个结果。

6. 如果每个结果都只有一分，则将备用列表中的实体词汇也检索一遍，扩充上联 ID。

7. 在做一遍操作5

8. 根据最终上联 ID，得到具体上联数据。返回该数据。


#### 生成下联

得到了所需要的上联数据，就要开始生成下联的工作了。该工作在 utils.py 的 `do_upload_image` 函数中继续完成。

调用 up2down_class.py 中的 `get_next` 函数，将上联数据作为参数一并传入。程序会用训练好的NLP模型进行推理（解码工作），对每一个候选上联，生成一个下联。

#### 合成对联

接下来，程序将每个上联和生成的下联合成一个以逗号分隔的对联形式，并以 json 的格式返回 ID 和前 N 个对联的结果（本案例中目前 code 写死为前三个）。

#### 重新创作

如果生成的下联用户不喜欢，可以更换对联。程序会根据当前上联，再次调用 `get_next` 函数，生成新的下联。该功能在函数 `do_modify_poetry` 中实现（ utils.py 文件）。

#### 合成图片

程序生成下联后，会显示几个备选对联（默认是3个）。用户可以选择自己喜欢的对联，并和上传的图片合成新的对联。具体处理过程如下：

1. 获取元数据信息

获取此次服务请求的 id 和选中的对联项。

2. 合成最终图片

包括：1) 用户上传图片 2) 程序背景图片3) 程序二维码图片 4) 微信图片 5) Logo图片 6) 对联内容

具体程序在 synthesis_2.py 文件中实现。合成好的图片保存在相应目录下，供前端应用查找并显示给用户。

### 前端应用

编写完后端程序，我们便可以和前端应用结合在一起，搭建起一个可以给用户使用的小程序。

#### 申请小程序账号

要开发微信小程序，需要先注册小程序公众号。微信小程序要求一个邮箱账号只能注册一个小程序，并且要填写真实完整的个人或公司信息。在安全性上，要求还是挺高的。
注册好小程序账号后，登录微信公众号的管理后台，下载微信小程序开发者工具，并进行开发设置。接下来便可以新建小程序项目，开发你的小程序了。具体教程见[微信小程序开发教程](https://developers.weixin.qq.com/miniprogram/dev/)。


#### 前端应用开发

微信小程序开发者工具提供了开发模板，开发者可以在不同模板文件中完成前端代码，类似网页开发。

#### 发布小程序

应用开发完成，开发者需要填写用户身份管理，上传代码，提交审核。审核通过，最终将小程序发布出去。这样，就完成了微信小程序的前端开发。

## 运行程序

我们在Azure上申请了一个VM，部署好我们的环境和代码，运行后端服务。

1. 登录VM，进入程序所在目录。
2. 若 VM 开启了 nginx 服务，则先关闭服务：
    ```
    sudo service nginx stop
    ```
3. 本案例使用 anaconda python 3.6 virtual environment，所以需要先激活 environment 。命令如下：
    ```
    activate py3_backend
    ```
4. 启动tmux，进入不同工作窗口进行工作：
    ```
    tmux new -s engine
    ```
5. 创建新的 tmux window，同时按下 ctrl+b，再按c
6. 在新窗口跳转到 syn_images 目录：`cd syn_images`
7. 启动 master 进程，并使用 https 访问的443默认端口：
    ```
    sudo python ~/code/Poetic-Image/tunnel.py 443
    ```
8. 重复步骤5
9.  在新窗口启动 slave server 1，用于生成下联的服务：
    ```
    python ~/code/Poetic-Image/http_server_py3.py 8001
    ```
10. 重复步骤5
11. 在新窗口启动 slave server 2，同样用于生成下联的服务：
    ```
    python ~/code/Poetic-Image/http_server_py3.py 8002
    ```
12. 重复步骤5
13. 在新窗口启动 slave server 3 ，用于合成对联和图片的服务：
    ```
    python ~/code/Poetic-Image/http_server_py3.py 8003
    ```

程序启动完毕。这时，在微信小程序端就可以使用对联服务啦。


# 作业和挑战

1. 程序复现

从 GitHub 上下载70万条[对联数据](https://github.com/wb14123/couplet-dataset/releases)（couplet.tar.gz 文件），按照上述教程进行数据预处理，并使用 Tensor2Tensor 库进行模型训练。

2. 增量改进

   1. 数据收集：在 GitHub 上收集更多对联数据，进行模型训练。

   2. 使用 [Fairseq](https://github.com/pytorch/fairseq) 库进行模型训练，并对比两种模型库库不同算法之间的训练效果。

   3. 自己编写用户交互程序，调用模型推理API，完成给出上联，自动生成下联的工作。

   4. 扩展对联生成程序，用于古诗、绝句等的自动生成。


