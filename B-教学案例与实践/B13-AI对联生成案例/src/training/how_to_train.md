# 环境配置

Ubuntu 16.04

Anaconda3 安装 python3.x 环境，或本机 Python3.x 环境

安装 training/train_requirements.txt 中所需要的 python packages


# 数据收集

1. 从 GitHub 网站下载 couplet v1.0 release 版本的数据，并解压。

    **GitHub网站**：https://github.com/wb14123/couplet-dataset/releases

    **文件名**：couplet.tar.gz

2. 下载 Tensor2Tensor 库 v1.2.9 版本
    ```
    git clone https://github.com/tensorflow/tensor2tensor.git
    git checkout v1.2.9
    ```


# 数据处理

Couplet 数据解压后，有两个文件夹，分别存放 train 和 test 的上下联数据。 In 为上联， out 为下联。在训练过程中，我们只需对 train 的数据进行预处理。

1. 创建 data 子目录。
2. 将 couplet 中 train 和 test 目录下的数据移动至 data 目录下，并重命名为：
    ```
    train.in.txt，train.out.txt，test.in.txt，test.out.txt。
    ```
3. 在 data 目录中，合并上下联文件并保存结果：
    ```
    cat train.in.txt train.out.txt > train.merge.txt
    ```
4. 在data 目录中，统计字表数目并保存字表：

    需要安装 subword-nmt 包：
    ```
    python -m pip install subword-nmt
    ```

    统计字数命令：
    ```
    subword-nmt get-vocab –input train.merge.txt –output train.merge.txt.vocab
    ```
5. 在 data 目录中，去掉字表文件中的汉字出现次数，只留字表：
    ```
    cat train.merge.txt.vocab | awk ‘{print $1}’ > merge.txt.vocab.clean
    ```
6. 将 merge.txt.vocab.clean 的前三行填充如下内容，并将字表字数加3：
    ```
    <pad>
    <EOS>
    <UNK>
    ```
7. 将 training/usr_dir 目录下的 merge_vocab.py 和 \_\_init\_\_.py 文件拷贝到 data 目录。
8. 修改 merge_vocab.py 中的内容：

    a. SRC_TRAIN_DATA 为训练集上联数据文件

    b. TGT_TRAIN_DATA 为训练集下联数据文件

    c. SRC_DEV_DATA 为测试集上联数据文件

    d. TGT_DEV_DATA 为测试集下联数据文件

    e. MERGE_VOCAB 为最终字表文件

    f. VOCAB_SIZE 为字表文件中字的个数


# 数据生成

1. 修改 data_gen.sh 脚本内容。
2. 运行 `bash data_gen.sh` 脚本，完成数据生成工作。


# 数据训练

1. 创建 output 目录存放训练结果。
2. 修改 train.sh 脚本内容，如 *train_steps*，*batch_size* 等为你想要的数值。
3. 运行 `bash train.sh` 脚本，完成数据处理工作。
4. 数据训练完成，则在 output 目录下生成训练数据。

    我设置的 batch_size 为 100000 ，所以最终需要的数据文件为：
    ```
    model.ckpt-100000.data-00000-of-00003
    model.ckpt-100000.data-00001-of-00003
    model.ckpt-100000.data-00002-of-00003
    model.ckpt-100000.index
    model.ckpt-100000.meta
    checkpoint
    ```
