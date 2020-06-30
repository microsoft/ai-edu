# 使用fairseq训练模型

在开始之前，请先确保已成功安装fairseq。
```
pip install fairseq
```

## 训练数据下载
我们可以在这个链接下载开源的对联数据： https://github.com/wb14123/couplet-dataset/releases

下载后，我们可以分别得到`train`和`test`目录下的上联文件`in.txt`和下联文件`out.txt`。

为了便于后续重新划分数据，我们先使用如下命令合并`train`和`test`目录下的数据：
```
cat test/in.txt train/in.txt > in.txt
cat test/out.txt train/out.txt > out.txt
```


如果您打算使用自己的数据集，请确保对联的数据分为上联和下联两个文件，用换行符`\n`分隔每条上联或下联数据，每个字以空格隔开。

## 划分数据集
在开始训练之前，我们需要将收集的训练数据分为训练集、验证集、测试集三部分。

由于训练数据比较大，我们这里使用98:1:1的比例划分训练集、验证集、测试集，并将上联文件分别命名为`train.up`、`valid.up`、`test.up`，下联文件命名为`train.down`、`valid.down`、`test.down`。

以下提供了两种划分数据集的方式供参考。
### Python脚本划分数据集
该方法需要确保已安装sklearn，如未安装可以执行命令```pip3 install -U scikit-learn```安装。

首先，将以下脚本保存为`split.py`，并与对联数据`in.txt`和`out.txt`放置于同一目录下。
```
from sklearn.model_selection import train_test_split
import sys

x_file = sys.argv[1]
y_file = sys.argv[2]

with open(x_file,'r') as f:
    X = f.read().splitlines()

with open(y_file,'r') as f:
    y = f.read().splitlines()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=1)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

def save(name,data):
    with open(name,'w') as f:
        for l in data:
            f.write(l+'\n')

save('./train.up',X_train)
save('./train.down',y_train)
save('./test.up',X_test)
save('./test.down',y_test)
save('./valid.up',X_val)
save('./valid.down',y_val)
```

然后，执行命令：

```
python split.py in.txt out.txt
```

脚本将使用98:1:1的比例划分训练集、验证集、测试集，自动生成对应的文件。

### Bash脚本划分
该方法利用Linux系统中的`wc`和`split`命令按照行数手动划分数据集，与上一种方法相比，该方法没有打乱数据集后进行划分。

具体步骤如下：

1. 执行以下命令统计数据集行数。
    ```
    wc -l in.txt
    ```
    得到以下输出：
    ```
    774491 in.txt
    ```
    由于数据集共计774491行，按照98：1：1的比例划分数据集，因此我们需要把数据分按759001、7745、7745行进行划分。

2. 执行以下命令划分出上联的训练集。
    ```
    split -l 759001 in.txt
    ```
    完成后数据会被划分为`xaa`和`xab`两个文件。`xaa`包含了759001行数据，`xab`包含了剩余的数据。

    执行以下命令将`xaa`重命名为`train.up`即可。
    ```
    mv xaa train.up
    ```

3. 执行以下命令继续划分上联的验证集和数据集。（如果出现overwrite error，可以重命名`xab`后重新执行）
    ```
    split -l 7745 xab
    ```
    划分完成后，将`xaa`命名为`test.up`，将`xab`命名为`valid.up`即可。命令如下:
    ```
    mv xaa test.up
    mv xab valid.up
    ```

4. 重复上述过程划分下联。注意划分顺序必须与上述过程保持一致，否则，将导致上联与下联数据不匹配。
    
    执行以下命令即可完成下联的划分。
    ```
    split -l 759001 out.txt
    mv xaa train.down
    split -l 7745 xab
    mv xaa test.down
    mv xab valid.down
    ```

## 数据预处理
在训练模型之前，我们需要将数据预处理成模型可以读取的二进制文件，并生成字典文件。

开始前，我们先统一文件路径。
```
HOME_DIR=$(cd `dirname $0`; pwd)
RAW_DATA_DIR=${HOME_DIR}/fairseq-data
PREPROCESSED_DATA_DIR=${HOME_DIR}/data-bin/couplet
MODEL_SAVE_DIR=${HOME_DIR}/output/couplet
```

* `RAW_DATA_DIR`为上述的所有训练数据的存放目录。上述步骤划分数据集后得到的所有文件均存放于此目录
* `PREPROCESSED_DATA_DIR`是预处理文件的输出目录
* `MODEL_SAVE_DIR`为训练模型结果的保存目录

接着执行以下脚本开始预处理数据。

```
fairseq-preprocess \
--source-lang up \
--target-lang down \
--trainpref ${RAW_DATA_DIR}/train \
--validpref ${RAW_DATA_DIR}/valid \
--testpref ${RAW_DATA_DIR}/test \
--destdir ${PREPROCESSED_DATA_DIR}
```

完成预处理后，生成的训练所需的二进制文件及上联和下联的字典文件都会保存在`PREPROCESSED_DATA_DIR`目录下。

## 模型训练

完成数据预处理后，执行以下脚本即可开始训练。
```
fairseq-train ${PREPROCESSED_DATA_DIR} \
--log-interval 100 \
--lr 0.25 \
--clip-norm 0.1 \
--dropout 0.2  \
--criterion label_smoothed_cross_entropy \
--save-dir ${MODEL_SAVE_DIR} \
-a lstm \
--max-tokens 4000 \
--max-epoch 100
```

其中，`-a`参数可以选择训练的模型，此处我们选择lstm进行训练。

更多的参数解释请参考[fairseq文档](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-train)。

训练完成后，模型文件会保存在`MODEL_SAVE_DIR`目录下。

## 模型推理

fairseq提供了两种模型推理的命令行工具，分别是fairseq-generate和fairseq-interactive。除此之外，我们还可以使用Python加载模型并完成推理。

### fairseq-generate
fairseq-generate的输入是二进制文件，会自动读取测试集的数据完成推理。

具体命令如下：
```
fairseq-generate ${PREPROCESSED_DATA_DIR} --path ${MODEL_SAVE_DIR}/checkpoint_best.pt --source-lang up --target-lang down
```

### fair-seq-interactive
fairseq-interactive提供了交互式命令行的方式推理，加载模型后，用户输入上联，模型将实时输出下联。

具体命令如下：
```
fairseq-interactive ${PREPROCESSED_DATA_DIR} --path ${MODEL_SAVE_DIR}/checkpoint_best.pt --source-lang up --target-lang down
```

更多参数请参考[fairseq文档](https://fairseq.readthedocs.io/en/latest/command_line_tools.html#fairseq-interactive)。

### 使用Python加载模型
为了搭建后端服务，我们可以使用Python加载模型，并完成推理。

下面以加载LSTM为例。

1. 引入模型

    ```
    from fairseq.models.lstm import LSTMModel
    ```

2. 读入模型文件
    
    我们将模型生成的checkpoint文件拷贝到`./checkpoints`目录下，将预处理阶段生成的两份字典文件（`dict.up.txt`和`dict.down.txt`）拷贝到`./checkpoints/dict`目录下。按如下代码即可读入文件：
    ```
    model = LSTMModel.from_pretrained('./checkpoints',\
    checkpoint_file='checkpoint_best.pt',\
    data_name_or_path="./dict")
    ```

    第一个参数为checkpoint文件所在目录，`checkpoint_file`为需要读入的checkpoint的文件名，`data_name_or_path`为字典文件所在的目录。需要注意的是，字典文件目录将以第一个参数为根目录，若路径有误会出现报错信息：`AttributeError: 'NoneType' object has no attribute 'split'`。

3. 推理

    此处需要注意输入的文字之间需用空格隔开。
    ```
    upper = '海内存知己'
    down = model.translate(' '.join(list(upper)))
    print(down) # 天 涯 若 比 邻
    ```

## 搭建Flask Web应用
1. 安装flask
    ```
    pip3 install flask
    ```

2. 搭建服务

    这一步我们将使用Python加载模型后，利用flask开启web服务。

    ```
    from flask import Flask
    from flask import request
    from fairseq.models.lstm import LSTMModel # 引入模型

    model = LSTMModel.from_pretrained('./checkpoints',\
        checkpoint_file='checkpoint_best.pt',\
        data_name_or_path="./dict") # 读入模型

    app = Flask(__name__)

    @app.route('/',methods=['GET'])
    def get_couplet_down():
        couplet_up = request.args.get('upper','')

        couplet_down = model.translate(' '.join(list(couplet_up))) # 模型推理

        couplet_down = couplet_down.replace(' ','')

        return couplet_up + "," + couplet_down

    ```
3. 启动服务

    在测试环境中，我们使用flask自带的web服务即可（注：生产环境应使用uwsgi+nginx部署，有兴趣的同学可以自行查阅资料）。

    使用以下两条命令：

    在Ubuntu下，
    ```
    export FLASK_APP=app.py
    python -m flask run
    ```
    在Windows下，
    ```
    set FLASK_APP=app.py
    python -m flask run
    ```
    此时，服务就启动啦。

    我们仅需向后端 http://127.0.0.1:5000/ 发起get请求，并带上上联参数upper，即可返回生成的对联到前端。

    请求示例: ```http://127.0.0.1:5000/?upper=海内存知己```

    返回结果: ```海内存知己，天涯若比邻```

## 模型对比

除了LSTM外，我们还使用了fairseq中内置的几组模型进行训练对比，包括CNN、transformer。具体可用模型可以参考[fairseq Models](https://fairseq.readthedocs.io/en/latest/models.html)。

我们对模型训练了30个epoch。

速度和模型大小对比：
| 模型名称 |推理时间 (100句) | 训练时间 | 模型大小 |
|---|---|---|---|
| LSTM |0.7s| 2h48m| 103.9M |
| CNN | 2.9s|28h42m|679.1M |
| Transformer|1.1s|4h05m|207.9M|

训练效果对比：
| 模型名称 | Valid Loss | BLEU4 | Perplexity |
|---|---|---|---|
| LSTM |3.99| 12.14| 17.00 |
| CNN | 4.39|15.69| 19.39 |
| Transformer|4.24|10.29|10.77|

注：Transformer为2个Encoder和2个Decoder。

可见，LSTM的模型较小，训练和推理速度较快，loss也更小。